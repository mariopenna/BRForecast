"""BRForecast — Dashboard Streamlit.

Dashboard interativo para previsao do Campeonato Brasileiro Serie A.
Combina ratings ELO, modelo Poisson/Dixon-Coles e simulacao Monte Carlo.

Uso:
    streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import sys

# Ensure src package is importable
sys.path.insert(0, os.path.dirname(__file__))

from data_processor import (
    load_simulation_results, load_team_stats, load_remaining_matches,
    load_current_table, load_elo_csv,
    compute_elo_data, compute_strengths, compute_match_cards,
    compute_upcoming_probs, run_whatif,
    get_zone, get_zone_color, get_zone_label, format_pct,
    verdict_color, ZONE_COLORS, TARGET_YEAR,
)

# =========================================================================
# Page config
# =========================================================================

st.set_page_config(
    page_title="BRForecast — Brasileirao Serie A",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# Sidebar navigation
# =========================================================================

st.sidebar.title("BRForecast")
st.sidebar.caption(f"Campeonato Brasileiro Serie A {TARGET_YEAR}")

SECTIONS = [
    "Probabilidades Gerais",
    "Power Rankings",
    "Heatmap de Posicoes",
    "Deep Dive por Time",
    "Visao Individual de Jogos",
    "Cenario What-if",
    "Modelo vs Odds",
]

section = st.sidebar.radio("Navegacao", SECTIONS)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dados:** FootyStats | "
    "**Modelo:** ELO + Poisson/Dixon-Coles + Monte Carlo"
)


# =========================================================================
# Section 1: Probabilidades Gerais
# =========================================================================

def section_probabilidades():
    st.header("Probabilidades Gerais")

    sim = load_simulation_results()
    stats = load_team_stats()

    # Merge pontos atuais
    table = load_current_table()
    sim = sim.merge(
        table[['team', 'pontos', 'jogos', 'posicao']],
        on='team', how='left',
    )

    # --- Tabela principal ---
    st.subheader("Classificacao e Probabilidades")

    # Preparar dados para exibicao
    display = sim[['team', 'posicao', 'pontos', 'jogos',
                    'p_titulo', 'p_libertadores', 'p_sulamericana',
                    'p_rebaixamento', 'pts_mean', 'pos_mean']].copy()
    display = display.sort_values('pos_mean').reset_index(drop=True)

    display.columns = [
        'Time', 'Pos', 'Pts', 'J',
        'Titulo', 'Libertadores', 'Sul-Americana',
        'Rebaixamento', 'Pts Esp.', 'Pos Esp.',
    ]

    # Formatacao
    for col in ['Titulo', 'Libertadores', 'Sul-Americana', 'Rebaixamento']:
        display[col] = display[col].apply(lambda x: f"{x:.1%}" if x >= 0.01 else ("< 1%" if x > 0 else "—"))

    display['Pts Esp.'] = display['Pts Esp.'].apply(lambda x: f"{x:.1f}")
    display['Pos Esp.'] = display['Pos Esp.'].apply(lambda x: f"{x:.1f}")
    display['Pos'] = display['Pos'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "")

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        height=740,
    )

    # --- Grafico de barras horizontais ---
    st.subheader("Probabilidades por Zona")

    sim_sorted = sim.sort_values('pos_mean')

    fig = go.Figure()

    # Rebaixamento (vermelho, direita)
    fig.add_trace(go.Bar(
        y=sim_sorted['team'],
        x=sim_sorted['p_rebaixamento'],
        name='Rebaixamento',
        orientation='h',
        marker_color=ZONE_COLORS['rebaixamento'],
        hovertemplate='%{y}: %{x:.1%}<extra>Rebaixamento</extra>',
    ))

    # Sul-Americana (azul)
    fig.add_trace(go.Bar(
        y=sim_sorted['team'],
        x=sim_sorted['p_sulamericana'],
        name='Sul-Americana',
        orientation='h',
        marker_color=ZONE_COLORS['sulamericana'],
        hovertemplate='%{y}: %{x:.1%}<extra>Sul-Americana</extra>',
    ))

    # Libertadores (verde)
    fig.add_trace(go.Bar(
        y=sim_sorted['team'],
        x=sim_sorted['p_libertadores'],
        name='Libertadores',
        orientation='h',
        marker_color=ZONE_COLORS['libertadores'],
        hovertemplate='%{y}: %{x:.1%}<extra>Libertadores</extra>',
    ))

    fig.update_layout(
        barmode='group',
        height=700,
        xaxis_title='Probabilidade',
        xaxis_tickformat='.0%',
        yaxis=dict(autorange='reversed'),
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=150),
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Section 2: Power Rankings
# =========================================================================

def section_rankings():
    st.header("Power Rankings — ELO")

    elo_df = load_elo_csv()
    serie_a_teams = load_current_table()['team'].tolist()

    # Filtrar para Serie A atual
    elo_serie_a = elo_df[elo_df['team'].isin(serie_a_teams)].copy()
    elo_serie_a = elo_serie_a.sort_values('elo_rating', ascending=False).reset_index(drop=True)

    # --- ELO evolution chart ---
    st.subheader("Evolucao de ELO na Temporada")

    elo_ratings, elo_history = compute_elo_data()

    # Filtrar historico para Serie A 2026 (e um pouco antes para contexto)
    from src.config import SERIE_A_IDS
    season_id = SERIE_A_IDS.get(TARGET_YEAR, 0)

    # Pegar historico recente (2025+2026 para contexto)
    hist = elo_history[
        (elo_history['season_year'] >= TARGET_YEAR - 1)
        & (elo_history['division'] == 'A')
        & (elo_history['team'].isin(serie_a_teams))
    ].copy()
    hist['date'] = pd.to_datetime(hist['date_unix'], unit='s')

    # Variacao nas ultimas 5 rodadas
    recent = elo_history[
        (elo_history['season_year'] == TARGET_YEAR)
        & (elo_history['division'] == 'A')
        & (elo_history['team'].isin(serie_a_teams))
    ]
    if len(recent) > 0:
        max_gw = recent['game_week'].max()
        gw_start = max(1, max_gw - 4)
        elo_at_start = {}
        elo_at_end = {}
        for team in serie_a_teams:
            t_hist = recent[recent['team'] == team].sort_values('game_week')
            before = t_hist[t_hist['game_week'] <= gw_start]
            if len(before) > 0:
                elo_at_start[team] = before.iloc[-1]['elo_after']
            elif len(t_hist) > 0:
                elo_at_start[team] = t_hist.iloc[0]['elo_before']
            after = t_hist[t_hist['game_week'] <= max_gw]
            if len(after) > 0:
                elo_at_end[team] = after.iloc[-1]['elo_after']

        elo_serie_a['variacao_5r'] = elo_serie_a['team'].apply(
            lambda t: elo_at_end.get(t, 0) - elo_at_start.get(t, 0)
        )
    else:
        elo_serie_a['variacao_5r'] = 0

    # --- Tabela de rankings ---
    rank_display = elo_serie_a[['team', 'elo_rating', 'variacao_5r']].copy()
    rank_display.index = range(1, len(rank_display) + 1)
    rank_display.columns = ['Time', 'ELO', 'Var. 5 rodadas']
    rank_display['ELO'] = rank_display['ELO'].apply(lambda x: f"{x:.0f}")
    rank_display['Var. 5 rodadas'] = rank_display['Var. 5 rodadas'].apply(
        lambda x: f"+{x:.0f}" if x > 0 else f"{x:.0f}"
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Ranking Atual")
        st.dataframe(rank_display, use_container_width=True, height=740)

    with col2:
        if len(hist) > 0:
            fig = go.Figure()
            for team in serie_a_teams:
                team_data = hist[hist['team'] == team].sort_values('date')
                if len(team_data) == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=team_data['date'],
                    y=team_data['elo_after'],
                    mode='lines',
                    name=team,
                    hovertemplate=(
                        '%{x|%d/%m/%Y}<br>ELO: %{y:.0f}'
                        '<extra>%{fullData.name}</extra>'
                    ),
                ))

            fig.update_layout(
                title=f"Evolucao de ELO — Serie A {TARGET_YEAR - 1}-{TARGET_YEAR}",
                xaxis_title="Data",
                yaxis_title="Rating ELO",
                hovermode='x unified',
                template='plotly_white',
                height=740,
                legend=dict(
                    orientation='v',
                    font=dict(size=10),
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados de historico ELO para a temporada atual.")


# =========================================================================
# Section 3: Heatmap de Posicoes
# =========================================================================

def section_heatmap():
    st.header("Heatmap de Posicoes Finais")
    st.caption("Probabilidade de cada time terminar em cada posicao (Monte Carlo)")

    sim = load_simulation_results()

    # Extrair colunas de posicao
    pos_cols = [f'pos_{i}' for i in range(1, 21)]
    teams = sim.sort_values('pos_mean')['team'].tolist()

    # Construir matriz
    matrix = np.zeros((len(teams), 20))
    for i, team in enumerate(teams):
        row = sim[sim['team'] == team].iloc[0]
        for j in range(20):
            matrix[i, j] = row[pos_cols[j]]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[str(p) for p in range(1, 21)],
        y=teams,
        colorscale=[
            [0.0, '#FFFFFF'],
            [0.05, '#FFF9C4'],
            [0.15, '#FFE082'],
            [0.30, '#FF8A65'],
            [0.50, '#E53935'],
            [1.0, '#B71C1C'],
        ],
        hovertemplate=(
            '%{y}<br>'
            'Posicao %{x}: %{z:.1%}'
            '<extra></extra>'
        ),
        colorbar=dict(title='Prob.', tickformat='.0%'),
    ))

    # Adicionar linhas divisorias de zonas
    for pos in [6.5, 12.5, 16.5]:
        fig.add_vline(
            x=pos - 0.5, line_dash="dash", line_color="black",
            line_width=1, opacity=0.5,
        )

    # Anotacoes de zona
    fig.add_annotation(x=2.5, y=-0.8, text="Libertadores",
                       showarrow=False, font=dict(color=ZONE_COLORS['libertadores'], size=11))
    fig.add_annotation(x=8.5, y=-0.8, text="Sul-Americana",
                       showarrow=False, font=dict(color=ZONE_COLORS['sulamericana'], size=11))
    fig.add_annotation(x=17.5, y=-0.8, text="Rebaixamento",
                       showarrow=False, font=dict(color=ZONE_COLORS['rebaixamento'], size=11))

    fig.update_layout(
        xaxis_title="Posicao Final",
        yaxis=dict(autorange='reversed'),
        height=700,
        template='plotly_white',
        margin=dict(l=150, b=80),
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Section 4: Deep Dive por Time
# =========================================================================

def section_deep_dive():
    st.header("Deep Dive por Time")

    sim = load_simulation_results()
    stats = load_team_stats()
    teams_sorted = sim.sort_values('pos_mean')['team'].tolist()

    selected = st.selectbox("Selecione o time", teams_sorted)

    team_sim = sim[sim['team'] == selected].iloc[0]
    team_stats = stats[stats['team'] == selected]
    if len(team_stats) > 0:
        team_stats = team_stats.iloc[0]
    else:
        team_stats = None

    # --- Metricas resumo ---
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Posicao Atual", f"{team_sim.get('posicao', team_sim['pos_mean']):.0f}"
                if 'posicao' not in team_sim.index
                else f"{team_sim['pos_mean']:.1f}")
    col1.metric("Pos. Esperada", f"{team_sim['pos_mean']:.1f}")
    col2.metric("Titulo", f"{team_sim['p_titulo']:.1%}")
    col3.metric("Libertadores", f"{team_sim['p_libertadores']:.1%}")
    col4.metric("Sul-Americana", f"{team_sim['p_sulamericana']:.1%}")
    col5.metric("Rebaixamento", f"{team_sim['p_rebaixamento']:.1%}")

    # --- Stats do time ---
    if team_stats is not None:
        st.markdown("---")
        scol1, scol2, scol3, scol4, scol5, scol6 = st.columns(6)
        scol1.metric("Jogos", f"{team_stats['jogos']:.0f}")
        scol2.metric("Pontos", f"{team_stats['pontos']:.0f}")
        scol3.metric("ELO", f"{team_stats['elo']:.0f}")
        scol4.metric("Ataque", f"{team_stats['attack']:.2f}")
        scol5.metric("Defesa", f"{team_stats['defense']:.2f}")
        scol6.metric("Pts Esperados", f"{team_sim['pts_mean']:.1f}")

    st.markdown("---")

    # --- Distribuicao de posicoes ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribuicao de Posicoes Finais")
        pos_cols = [f'pos_{i}' for i in range(1, 21)]
        pos_probs = [team_sim[col] for col in pos_cols]
        colors = [get_zone_color(i + 1) for i in range(20)]

        fig_pos = go.Figure(go.Bar(
            x=list(range(1, 21)),
            y=pos_probs,
            marker_color=colors,
            hovertemplate='Posicao %{x}: %{y:.1%}<extra></extra>',
        ))
        fig_pos.update_layout(
            xaxis_title="Posicao",
            yaxis_title="Probabilidade",
            yaxis_tickformat='.0%',
            template='plotly_white',
            height=350,
            xaxis=dict(dtick=1),
        )
        st.plotly_chart(fig_pos, use_container_width=True)

    with col_right:
        st.subheader("Pontos Esperados")
        # Box plot-like visualization from percentiles
        fig_pts = go.Figure()

        fig_pts.add_trace(go.Box(
            q1=[team_sim['pts_p10']],
            median=[team_sim['pts_median']],
            q3=[team_sim['pts_p90']],
            lowerfence=[team_sim['pts_min']],
            upperfence=[team_sim['pts_max']],
            mean=[team_sim['pts_mean']],
            name=selected,
            marker_color=get_zone_color(int(team_sim['pos_mean'])),
            boxmean=True,
        ))

        fig_pts.update_layout(
            yaxis_title="Pontos",
            template='plotly_white',
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig_pts, use_container_width=True)

    # --- Evolucao de ELO na temporada ---
    st.subheader("Evolucao de ELO na Temporada")
    _, elo_history = compute_elo_data()

    team_hist = elo_history[
        (elo_history['team'] == selected)
        & (elo_history['season_year'] >= TARGET_YEAR - 1)
        & (elo_history['division'] == 'A')
    ].copy()

    if len(team_hist) > 0:
        team_hist['date'] = pd.to_datetime(team_hist['date_unix'], unit='s')
        team_hist = team_hist.sort_values('date')

        fig_elo = go.Figure()
        fig_elo.add_trace(go.Scatter(
            x=team_hist['date'],
            y=team_hist['elo_after'],
            mode='lines+markers',
            name=selected,
            marker=dict(size=5),
            line=dict(width=2),
            hovertemplate=(
                '%{x|%d/%m/%Y}<br>'
                'ELO: %{y:.0f}<br>'
                '<extra></extra>'
            ),
        ))

        # Adicionar pontos de resultado (cor por V/E/D)
        result_colors = {'W': '#2ECC71', 'D': '#F39C12', 'L': '#E74C3C'}
        for result, color in result_colors.items():
            subset = team_hist[team_hist['result'] == result]
            fig_elo.add_trace(go.Scatter(
                x=subset['date'],
                y=subset['elo_after'],
                mode='markers',
                name={'W': 'Vitoria', 'D': 'Empate', 'L': 'Derrota'}[result],
                marker=dict(size=8, color=color),
                hovertemplate=(
                    '%{x|%d/%m/%Y}<br>'
                    'ELO: %{y:.0f}<br>'
                    f'vs {result}<br>'
                    '<extra></extra>'
                ),
            ))

        fig_elo.update_layout(
            xaxis_title="Data",
            yaxis_title="Rating ELO",
            template='plotly_white',
            height=350,
        )
        st.plotly_chart(fig_elo, use_container_width=True)
    else:
        st.info("Sem historico ELO para a temporada atual.")

    # --- Proximos jogos ---
    st.subheader("Proximos Jogos")
    upcoming = compute_upcoming_probs()
    team_matches = upcoming[
        (upcoming['mandante'] == selected) | (upcoming['visitante'] == selected)
    ].head(10)

    if len(team_matches) > 0:
        rows = []
        for _, m in team_matches.iterrows():
            is_home = m['mandante'] == selected
            opponent = m['visitante'] if is_home else m['mandante']
            mando = "Casa" if is_home else "Fora"
            p_win = m['p_home'] if is_home else m['p_away']
            p_lose = m['p_away'] if is_home else m['p_home']

            rows.append({
                'Rodada': f"{m['rodada']:.0f}",
                'Adversario': opponent,
                'Mando': mando,
                f'P(V {selected})': f"{p_win:.1%}",
                'P(E)': f"{m['p_draw']:.1%}",
                f'P(D {selected})': f"{p_lose:.1%}",
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("Nenhum jogo restante.")


# =========================================================================
# Section 5: Visao Individual de Jogos
# =========================================================================

def section_match_view():
    st.header("Visao Individual de Jogos")

    cards = compute_match_cards()

    if len(cards) == 0:
        st.warning("Nenhum jogo realizado na temporada atual.")
        return

    # Construir labels para o dropdown
    cards_sorted = cards.sort_values('game_week', ascending=False)
    labels = []
    for _, c in cards_sorted.iterrows():
        gw = c['game_week']
        label = f"R{gw:.0f} — {c['home']} {c['goals_home']:.0f}x{c['goals_away']:.0f} {c['away']}"
        labels.append(label)

    selected_label = st.selectbox("Selecione o jogo", labels)
    selected_idx = labels.index(selected_label)
    card = cards_sorted.iloc[selected_idx]

    # --- Ficha do jogo ---
    st.markdown("---")

    # Header com placar
    header_col1, header_col2, header_col3 = st.columns([2, 1, 2])
    with header_col1:
        st.markdown(f"### {card['home']}")
    with header_col2:
        st.markdown(
            f"<h2 style='text-align:center'>"
            f"{card['goals_home']:.0f} x {card['goals_away']:.0f}"
            f"</h2>",
            unsafe_allow_html=True,
        )
    with header_col3:
        st.markdown(f"### {card['away']}")

    st.markdown(
        f"**Rodada {card['game_week']:.0f}** | {card.get('date', '')} | "
        f"**{card['result_text']}**"
    )

    # Veredicto
    v_color = verdict_color(card['verdict'])
    st.markdown(
        f"<span style='background-color:{v_color}; color:white; "
        f"padding:4px 12px; border-radius:4px; font-weight:bold'>"
        f"{card['verdict']}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # xG comparison
    if card['has_xg']:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("xG")
            fig_xg = go.Figure()
            fig_xg.add_trace(go.Bar(
                x=[card['home'], card['away']],
                y=[card['xg_home'], card['xg_away']],
                marker_color=['#3498DB', '#E74C3C'],
                name='xG',
                text=[f"{card['xg_home']:.2f}", f"{card['xg_away']:.2f}"],
                textposition='outside',
            ))
            fig_xg.add_trace(go.Bar(
                x=[card['home'], card['away']],
                y=[card['goals_home'], card['goals_away']],
                marker_color=['#2980B9', '#C0392B'],
                name='Gols',
                text=[f"{card['goals_home']:.0f}", f"{card['goals_away']:.0f}"],
                textposition='outside',
                opacity=0.6,
            ))
            fig_xg.update_layout(
                barmode='group', height=300, template='plotly_white',
                showlegend=True,
            )
            st.plotly_chart(fig_xg, use_container_width=True)

        with col2:
            st.subheader("Modelo Poisson")
            st.metric("P(V Casa)", f"{card['p_home_win']:.1%}")
            st.metric("P(Empate)", f"{card['p_draw']:.1%}")
            st.metric("P(V Fora)", f"{card['p_away_win']:.1%}")
            st.metric("P(Placar exato)", f"{card['p_exact_score']:.1%}")

        with col3:
            if card['has_odds']:
                st.subheader("Odds vs Modelo")

                comp_data = pd.DataFrame({
                    'Resultado': ['Casa', 'Empate', 'Fora'],
                    'Modelo': [card['p_home_win'], card['p_draw'], card['p_away_win']],
                    'Odds': [card['odds_p_home'], card['odds_p_draw'], card['odds_p_away']],
                })

                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    x=comp_data['Resultado'], y=comp_data['Modelo'],
                    name='Modelo', marker_color='#3498DB',
                    text=comp_data['Modelo'].apply(lambda x: f"{x:.1%}"),
                    textposition='outside',
                ))
                fig_comp.add_trace(go.Bar(
                    x=comp_data['Resultado'], y=comp_data['Odds'],
                    name='Odds', marker_color='#F39C12',
                    text=comp_data['Odds'].apply(lambda x: f"{x:.1%}"),
                    textposition='outside',
                ))
                fig_comp.update_layout(
                    barmode='group', height=300, template='plotly_white',
                    yaxis_tickformat='.0%',
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.subheader("Odds")
                st.info("Odds nao disponiveis para este jogo.")

    # --- Stats extras ---
    if card.get('has_extra_stats'):
        st.subheader("Estatisticas da Partida")
        stat_names = {
            'shots': 'Chutes',
            'shots_on_target': 'Chutes no alvo',
            'possession': 'Posse (%)',
            'corners': 'Escanteios',
            'dangerous_attacks': 'Ataques perigosos',
            'fouls': 'Faltas',
        }
        rows = []
        for key, label in stat_names.items():
            h_val = card.get(f'{key}_home')
            a_val = card.get(f'{key}_away')
            if pd.notna(h_val) and pd.notna(a_val):
                rows.append({
                    card['home']: f"{h_val:.0f}",
                    'Estatistica': label,
                    card['away']: f"{a_val:.0f}",
                })

        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# =========================================================================
# Section 6: Cenario What-if
# =========================================================================

def section_whatif():
    st.header("Cenario What-if")
    st.caption("Fixe resultados de jogos futuros e veja como as probabilidades mudam.")

    # Session state para jogos fixados
    if 'fixed_results' not in st.session_state:
        st.session_state.fixed_results = {}
    if 'whatif_results' not in st.session_state:
        st.session_state.whatif_results = None

    remaining = load_remaining_matches()

    # --- Interface para fixar resultados ---
    st.subheader("Fixar Resultados")

    match_labels = []
    match_keys = []
    for _, m in remaining.iterrows():
        label = f"R{m['rodada']:.0f} — {m['mandante']} vs {m['visitante']}"
        match_labels.append(label)
        match_keys.append((m['mandante'], m['visitante']))

    col1, col2, col3 = st.columns([3, 2, 1])

    with col1:
        sel_match_idx = st.selectbox(
            "Jogo", range(len(match_labels)),
            format_func=lambda i: match_labels[i],
            key='whatif_match',
        )

    with col2:
        sel_result = st.radio(
            "Resultado",
            ["Vitoria Mandante", "Empate", "Vitoria Visitante"],
            horizontal=True,
            key='whatif_result',
        )

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Adicionar", type="primary"):
            key = match_keys[sel_match_idx]
            if sel_result == "Vitoria Mandante":
                goals = (2, 0)
            elif sel_result == "Empate":
                goals = (1, 1)
            else:
                goals = (0, 2)
            st.session_state.fixed_results[key] = goals

    # --- Lista de jogos fixados ---
    if st.session_state.fixed_results:
        st.subheader(f"Jogos Fixados ({len(st.session_state.fixed_results)})")

        for key, goals in list(st.session_state.fixed_results.items()):
            home, away = key
            hg, ag = goals

            if hg > ag:
                result_text = f"V {home}"
            elif hg == ag:
                result_text = "Empate"
            else:
                result_text = f"V {away}"

            cols = st.columns([4, 1])
            with cols[0]:
                st.text(f"{home} {hg}x{ag} {away} ({result_text})")
            with cols[1]:
                if st.button("Remover", key=f"rm_{home}_{away}"):
                    del st.session_state.fixed_results[key]
                    st.session_state.whatif_results = None
                    st.rerun()

        col_sim, col_clear = st.columns([1, 1])
        with col_sim:
            if st.button("Simular", type="primary"):
                with st.spinner("Rodando 5.000 simulacoes..."):
                    st.session_state.whatif_results = run_whatif(
                        TARGET_YEAR, st.session_state.fixed_results, n_sims=5000,
                    )
        with col_clear:
            if st.button("Limpar tudo"):
                st.session_state.fixed_results = {}
                st.session_state.whatif_results = None
                st.rerun()
    else:
        st.info("Nenhum jogo fixado. Use o formulario acima para adicionar resultados.")

    # --- Resultados comparativos ---
    if st.session_state.whatif_results is not None:
        st.markdown("---")
        st.subheader("Comparacao: Cenario Base vs What-if")

        base = load_simulation_results()
        whatif = st.session_state.whatif_results

        comp = base[['team', 'p_titulo', 'p_libertadores', 'p_rebaixamento', 'pos_mean']].merge(
            whatif[['team', 'p_titulo', 'p_libertadores', 'p_rebaixamento', 'pos_mean']],
            on='team', suffixes=('_base', '_whatif'),
        )
        comp['delta_titulo'] = comp['p_titulo_whatif'] - comp['p_titulo_base']
        comp['delta_liberta'] = comp['p_libertadores_whatif'] - comp['p_libertadores_base']
        comp['delta_rebx'] = comp['p_rebaixamento_whatif'] - comp['p_rebaixamento_base']
        comp = comp.sort_values('pos_mean_whatif')

        # Tabela comparativa
        display = []
        for _, r in comp.iterrows():
            display.append({
                'Time': r['team'],
                'Titulo (base)': f"{r['p_titulo_base']:.1%}",
                'Titulo (w-if)': f"{r['p_titulo_whatif']:.1%}",
                'Delta Tit.': f"{r['delta_titulo']:+.1%}",
                'Liberta (base)': f"{r['p_libertadores_base']:.1%}",
                'Liberta (w-if)': f"{r['p_libertadores_whatif']:.1%}",
                'Rebx (base)': f"{r['p_rebaixamento_base']:.1%}",
                'Rebx (w-if)': f"{r['p_rebaixamento_whatif']:.1%}",
            })

        st.dataframe(pd.DataFrame(display), hide_index=True, use_container_width=True)

        # Grafico de maiores mudancas
        st.subheader("Times Mais Afetados")
        comp['total_delta'] = (comp['delta_titulo'].abs()
                               + comp['delta_liberta'].abs()
                               + comp['delta_rebx'].abs())
        top_affected = comp.nlargest(10, 'total_delta')

        fig_delta = go.Figure()
        fig_delta.add_trace(go.Bar(
            x=top_affected['team'],
            y=top_affected['delta_titulo'],
            name='Delta Titulo',
            marker_color=ZONE_COLORS['libertadores'],
        ))
        fig_delta.add_trace(go.Bar(
            x=top_affected['team'],
            y=top_affected['delta_liberta'],
            name='Delta Libertadores',
            marker_color='#27AE60',
        ))
        fig_delta.add_trace(go.Bar(
            x=top_affected['team'],
            y=top_affected['delta_rebx'],
            name='Delta Rebaixamento',
            marker_color=ZONE_COLORS['rebaixamento'],
        ))
        fig_delta.update_layout(
            barmode='group',
            yaxis_tickformat='+.1%',
            yaxis_title='Variacao de Probabilidade',
            template='plotly_white',
            height=400,
        )
        st.plotly_chart(fig_delta, use_container_width=True)


# =========================================================================
# Section 7: Modelo vs Odds
# =========================================================================

def section_odds_comparison():
    st.header("Modelo vs Odds")

    cards = compute_match_cards()

    if len(cards) == 0:
        st.warning("Nenhum jogo realizado para comparacao.")
        return

    # Filtrar jogos com odds
    with_odds = cards[cards['has_odds']].copy()
    if len(with_odds) == 0:
        st.warning("Nenhum jogo com odds disponivel.")
        return

    # --- Brier Score comparativo ---
    st.subheader("Metricas de Qualidade")

    # Calcular Brier Scores
    brier_model = 0.0
    brier_odds = 0.0
    log_loss_model = 0.0
    log_loss_odds = 0.0
    n = 0

    for _, c in with_odds.iterrows():
        result = c['result']
        if result == 'H':
            actual = (1, 0, 0)
        elif result == 'D':
            actual = (0, 1, 0)
        else:
            actual = (0, 0, 1)

        # Modelo
        brier_model += ((c['p_home_win'] - actual[0])**2
                        + (c['p_draw'] - actual[1])**2
                        + (c['p_away_win'] - actual[2])**2)
        p_actual_model = (c['p_home_win'] * actual[0]
                          + c['p_draw'] * actual[1]
                          + c['p_away_win'] * actual[2])
        log_loss_model += -np.log(max(p_actual_model, 1e-10))

        # Odds
        brier_odds += ((c['odds_p_home'] - actual[0])**2
                       + (c['odds_p_draw'] - actual[1])**2
                       + (c['odds_p_away'] - actual[2])**2)
        p_actual_odds = (c['odds_p_home'] * actual[0]
                         + c['odds_p_draw'] * actual[1]
                         + c['odds_p_away'] * actual[2])
        log_loss_odds += -np.log(max(p_actual_odds, 1e-10))

        n += 1

    brier_model /= n
    brier_odds /= n
    log_loss_model /= n
    log_loss_odds /= n

    col1, col2, col3 = st.columns(3)
    col1.metric("Jogos analisados", n)
    col2.metric("Brier Score", "", "")
    col3.metric("Log-Loss", "", "")

    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(
        x=['Brier Score', 'Log-Loss'],
        y=[brier_model, log_loss_model],
        name='Modelo',
        marker_color='#3498DB',
        text=[f"{brier_model:.4f}", f"{log_loss_model:.4f}"],
        textposition='outside',
    ))
    fig_metrics.add_trace(go.Bar(
        x=['Brier Score', 'Log-Loss'],
        y=[brier_odds, log_loss_odds],
        name='Odds',
        marker_color='#F39C12',
        text=[f"{brier_odds:.4f}", f"{log_loss_odds:.4f}"],
        textposition='outside',
    ))
    fig_metrics.update_layout(
        barmode='group', template='plotly_white', height=350,
        yaxis_title='Score (menor = melhor)',
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    # --- Grafico de calibracao ---
    st.subheader("Calibracao do Modelo")
    st.caption("Diagonal = calibracao perfeita. Acima = sub-confiante. Abaixo = sobre-confiante.")

    # Calcular calibracao para modelo e odds
    def calc_calibration(probs, actuals, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_actuals = []
        bin_counts = []
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_actuals.append(actuals[mask].mean())
                bin_counts.append(mask.sum())
        return bin_centers, bin_actuals, bin_counts

    # Coletar todas as previsoes (home/draw/away como eventos separados)
    all_model_probs = []
    all_odds_probs = []
    all_actuals = []

    for _, c in with_odds.iterrows():
        result = c['result']
        for outcome, m_prob, o_prob in [
            ('H', c['p_home_win'], c['odds_p_home']),
            ('D', c['p_draw'], c['odds_p_draw']),
            ('A', c['p_away_win'], c['odds_p_away']),
        ]:
            all_model_probs.append(m_prob)
            all_odds_probs.append(o_prob)
            all_actuals.append(1.0 if result == outcome else 0.0)

    all_model_probs = np.array(all_model_probs)
    all_odds_probs = np.array(all_odds_probs)
    all_actuals = np.array(all_actuals)

    m_centers, m_actuals, m_counts = calc_calibration(all_model_probs, all_actuals)
    o_centers, o_actuals, o_counts = calc_calibration(all_odds_probs, all_actuals)

    fig_cal = go.Figure()

    # Diagonal perfeita
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Perfeito',
        showlegend=True,
    ))

    fig_cal.add_trace(go.Scatter(
        x=m_centers, y=m_actuals,
        mode='lines+markers',
        name='Modelo',
        marker=dict(size=[max(5, min(15, c/3)) for c in m_counts], color='#3498DB'),
        hovertemplate='Previsto: %{x:.0%}<br>Real: %{y:.0%}<extra>Modelo</extra>',
    ))

    fig_cal.add_trace(go.Scatter(
        x=o_centers, y=o_actuals,
        mode='lines+markers',
        name='Odds',
        marker=dict(size=[max(5, min(15, c/3)) for c in o_counts], color='#F39C12'),
        hovertemplate='Previsto: %{x:.0%}<br>Real: %{y:.0%}<extra>Odds</extra>',
    ))

    fig_cal.update_layout(
        xaxis_title='Probabilidade Prevista',
        yaxis_title='Frequencia Real',
        xaxis_tickformat='.0%',
        yaxis_tickformat='.0%',
        template='plotly_white',
        height=450,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # --- Maiores divergencias ---
    st.subheader("Maiores Divergencias Modelo vs Odds")

    with_odds_copy = with_odds.copy()
    with_odds_copy['div_home'] = abs(with_odds_copy['p_home_win'] - with_odds_copy['odds_p_home'])
    with_odds_copy['div_draw'] = abs(with_odds_copy['p_draw'] - with_odds_copy['odds_p_draw'])
    with_odds_copy['div_away'] = abs(with_odds_copy['p_away_win'] - with_odds_copy['odds_p_away'])
    with_odds_copy['max_div'] = with_odds_copy[['div_home', 'div_draw', 'div_away']].max(axis=1)

    top_div = with_odds_copy.nlargest(20, 'max_div')

    display_div = []
    for _, c in top_div.iterrows():
        placar = f"{c['goals_home']:.0f}x{c['goals_away']:.0f}"
        display_div.append({
            'Rodada': f"{c['game_week']:.0f}",
            'Jogo': f"{c['home']} vs {c['away']}",
            'Placar': placar,
            'P(H) Modelo': f"{c['p_home_win']:.1%}",
            'P(H) Odds': f"{c['odds_p_home']:.1%}",
            'P(D) Modelo': f"{c['p_draw']:.1%}",
            'P(D) Odds': f"{c['odds_p_draw']:.1%}",
            'P(A) Modelo': f"{c['p_away_win']:.1%}",
            'P(A) Odds': f"{c['odds_p_away']:.1%}",
            'Max Div.': f"{c['max_div']:.1%}",
        })

    st.dataframe(pd.DataFrame(display_div), hide_index=True, use_container_width=True)


# =========================================================================
# Router
# =========================================================================

SECTION_FUNCS = {
    "Probabilidades Gerais": section_probabilidades,
    "Power Rankings": section_rankings,
    "Heatmap de Posicoes": section_heatmap,
    "Deep Dive por Time": section_deep_dive,
    "Visao Individual de Jogos": section_match_view,
    "Cenario What-if": section_whatif,
    "Modelo vs Odds": section_odds_comparison,
}

SECTION_FUNCS[section]()
