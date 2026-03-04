"""BRForecast — Sistema ELO para o Campeonato Brasileiro.

Usa Séries A, B e C dos últimos 5 anos completos + 2026.
- Série A pesa mais, Série C pesa menos (K_BASE_DIVISION)
- Temporadas mais recentes têm K maior (K_PROGRESSIVE)
- K efetivo = K_PROGRESSIVE[ano] × K_BASE_DIVISION[divisão]
- Rating inicial depende da divisão onde o time aparece pela primeira vez
"""

import sqlite3
import pandas as pd
import numpy as np
import os
from src.config import (
    DB_PATH, RATING_INICIAL, HFA, DATA_DIR,
    SERIE_A_IDS, SERIE_B_IDS, SERIE_C_IDS,
    K_PROGRESSIVE, K_BASE_DIVISION, ELO_WINDOW_START, TARGET_YEAR,
    ELO_SEASON_REGRESSION,
)


def load_historical_matches(db_path=DB_PATH):
    """Carrega partidas das Séries A, B e C na janela, ordenadas cronologicamente."""
    all_ids = {}
    for year, sid in SERIE_A_IDS.items():
        if year >= ELO_WINDOW_START:
            all_ids[sid] = ("A", year)
    for year, sid in SERIE_B_IDS.items():
        if year >= ELO_WINDOW_START:
            all_ids[sid] = ("B", year)
    for year, sid in SERIE_C_IDS.items():
        if year >= ELO_WINDOW_START:
            all_ids[sid] = ("C", year)

    season_ids = tuple(all_ids.keys())

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"""
        SELECT m.id, m.competition_id, l.year as season_year,
               m.homeID, m.awayID, m.home_name, m.away_name,
               m.homeGoalCount, m.awayGoalCount,
               m.date_unix, m.status, m.game_week
        FROM matches m
        JOIN leagues l ON m.competition_id = l.id
        WHERE m.competition_id IN {season_ids}
          AND m.status = 'complete'
        ORDER BY m.date_unix
    """, conn)
    conn.close()

    # Mapear divisão para cada partida
    df['division'] = df['competition_id'].map(lambda x: all_ids.get(x, ("?", 0))[0])

    return df


def expected_score(rating_a, rating_b, hfa=0):
    """Calcula o score esperado de A contra B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - (rating_a + hfa)) / 400.0))


def get_match_result(home_goals, away_goals):
    """Retorna (score_home, score_away) baseado no resultado."""
    if home_goals > away_goals:
        return (1.0, 0.0)
    elif home_goals < away_goals:
        return (0.0, 1.0)
    else:
        return (0.5, 0.5)


def update_rating(rating, score, expected, k):
    """Atualiza o rating ELO."""
    return rating + k * (score - expected)


def _get_effective_k(season_year, division):
    """K efetivo = K_PROGRESSIVE[ano] × K_BASE_DIVISION[divisão]."""
    k_time = K_PROGRESSIVE.get(season_year, 20)
    k_div = K_BASE_DIVISION.get(division, 0.5)
    return k_time * k_div


def _detect_first_division(matches_df):
    """Detecta em qual divisão cada time aparece pela primeira vez na janela.

    Retorna dict {team_name: 'A'|'B'|'C'}.
    """
    first_appearance = {}
    for _, match in matches_df.iterrows():
        for team in [match['home_name'], match['away_name']]:
            if team not in first_appearance:
                first_appearance[team] = match['division']
    return first_appearance


def calculate_all_elos(matches_df, hfa=HFA,
                       season_regression=ELO_SEASON_REGRESSION):
    """Calcula ELO iterando partida a partida (A+B+C, janela completa).

    Quando season_regression > 0, aplica regressao a media da divisao na
    fronteira de cada temporada:
        new_elo = elo * (1 - regression) + RATING_INICIAL[div] * regression

    A regressao usa a media da ultima divisao em que o time jogou,
    evitando que promovidos da B sejam superestimados na A.

    Retorna:
        ratings: dict {team_name: current_elo}
        history_df: DataFrame com histórico completo
    """
    first_div = _detect_first_division(matches_df)
    ratings = {}
    team_last_div = {}  # tracks last division each team played in
    history = []
    current_max_season = 0

    for _, match in matches_df.iterrows():
        home = match['home_name']
        away = match['away_name']
        season = match['season_year']
        division = match['division']

        # Regressao a media da divisao na fronteira de temporada
        if (season_regression > 0
                and season > current_max_season
                and current_max_season > 0):
            for team in ratings:
                div_mean = RATING_INICIAL.get(
                    team_last_div.get(team, "B"), 1350,
                )
                ratings[team] = (ratings[team] * (1 - season_regression)
                                 + div_mean * season_regression)
        if season > current_max_season:
            current_max_season = season

        # K efetivo para esta partida
        k = _get_effective_k(season, division)

        # Times novos: rating inicial pela divisão de primeira aparição
        if home not in ratings:
            ratings[home] = RATING_INICIAL.get(first_div.get(home, "B"), 1350)
        if away not in ratings:
            ratings[away] = RATING_INICIAL.get(first_div.get(away, "B"), 1350)

        # Track last division for regression target
        team_last_div[home] = division
        team_last_div[away] = division

        elo_home_before = ratings[home]
        elo_away_before = ratings[away]

        exp_home = expected_score(elo_home_before, elo_away_before, hfa=hfa)
        exp_away = 1.0 - exp_home

        score_home, score_away = get_match_result(
            match['homeGoalCount'], match['awayGoalCount']
        )

        ratings[home] = update_rating(elo_home_before, score_home, exp_home, k)
        ratings[away] = update_rating(elo_away_before, score_away, exp_away, k)

        history.append({
            'date_unix': match['date_unix'],
            'season_year': season,
            'division': division,
            'game_week': match['game_week'],
            'team': home,
            'elo_before': elo_home_before,
            'elo_after': ratings[home],
            'opponent': away,
            'is_home': True,
            'goals_for': match['homeGoalCount'],
            'goals_against': match['awayGoalCount'],
            'result': 'W' if score_home == 1.0 else ('D' if score_home == 0.5 else 'L'),
            'k_used': k,
        })
        history.append({
            'date_unix': match['date_unix'],
            'season_year': season,
            'division': division,
            'game_week': match['game_week'],
            'team': away,
            'elo_before': elo_away_before,
            'elo_after': ratings[away],
            'opponent': home,
            'is_home': False,
            'goals_for': match['awayGoalCount'],
            'goals_against': match['homeGoalCount'],
            'result': 'W' if score_away == 1.0 else ('D' if score_away == 0.5 else 'L'),
            'k_used': k,
        })

    history_df = pd.DataFrame(history)
    return ratings, history_df


def calibrate_hfa(matches_df, hfa_range=range(0, 130, 5)):
    """Grid search para encontrar o HFA que minimiza o Brier Score.

    Para cada valor de HFA:
      1. Recalcula ELOs com esse HFA
      2. Para cada partida, usa o ELO *antes* da atualização para prever o resultado
      3. Calcula Brier Score multiclasse (Home/Draw/Away)

    Retorna: (hfa_otimo, resultados_df com colunas [hfa, brier_score])
    """
    results = []

    for hfa in hfa_range:
        # Recalcular ELOs com este HFA — e coletar previsões in-sample
        first_div = _detect_first_division(matches_df)
        ratings = {}
        brier_sum = 0.0
        n = 0

        for _, match in matches_df.iterrows():
            home = match['home_name']
            away = match['away_name']
            season = match['season_year']
            division = match['division']
            k = _get_effective_k(season, division)

            if home not in ratings:
                ratings[home] = RATING_INICIAL.get(first_div.get(home, "B"), 1350)
            if away not in ratings:
                ratings[away] = RATING_INICIAL.get(first_div.get(away, "B"), 1350)

            # Prever ANTES de atualizar
            exp_home = expected_score(ratings[home], ratings[away], hfa=hfa)

            # Probabilidades tripartidas com taxa de empate constante histórica (~27%)
            draw_rate = 0.27
            p_home = exp_home * (1 - draw_rate)
            p_draw = draw_rate
            p_away = (1 - exp_home) * (1 - draw_rate)

            # Resultado real
            hg, ag = match['homeGoalCount'], match['awayGoalCount']
            if hg > ag:
                actual = (1, 0, 0)
            elif hg == ag:
                actual = (0, 1, 0)
            else:
                actual = (0, 0, 1)

            # Brier Score multiclasse
            brier_sum += (p_home - actual[0])**2 + (p_draw - actual[1])**2 + (p_away - actual[2])**2
            n += 1

            # Atualizar ELO
            score_home, score_away = get_match_result(hg, ag)
            exp_h = expected_score(ratings[home], ratings[away], hfa=hfa)
            ratings[home] = update_rating(ratings[home], score_home, exp_h, k)
            ratings[away] = update_rating(ratings[away], 1.0 - score_home, 1.0 - exp_h, k)

        brier = brier_sum / n
        results.append({'hfa': hfa, 'brier_score': brier, 'n_matches': n})

    results_df = pd.DataFrame(results)
    best_idx = results_df['brier_score'].idxmin()
    hfa_otimo = int(results_df.loc[best_idx, 'hfa'])

    return hfa_otimo, results_df


def analyze_draw_rate(matches_df, hfa):
    """Analisa taxa de empates por faixa de diferença de ELO.

    Recalcula ELOs com o HFA dado e para cada partida registra:
    - |elo_home + hfa - elo_away|
    - se foi empate ou não

    Agrupa em faixas e calcula taxa de empates em cada uma.
    Retorna: draw_analysis_df com colunas [faixa, n_jogos, empates, draw_rate]
    """
    first_div = _detect_first_division(matches_df)
    ratings = {}
    records = []

    for _, match in matches_df.iterrows():
        home = match['home_name']
        away = match['away_name']
        season = match['season_year']
        division = match['division']
        k = _get_effective_k(season, division)

        if home not in ratings:
            ratings[home] = RATING_INICIAL.get(first_div.get(home, "B"), 1350)
        if away not in ratings:
            ratings[away] = RATING_INICIAL.get(first_div.get(away, "B"), 1350)

        elo_diff = abs((ratings[home] + hfa) - ratings[away])
        is_draw = 1 if match['homeGoalCount'] == match['awayGoalCount'] else 0

        records.append({'elo_diff': elo_diff, 'is_draw': is_draw})

        # Atualizar ELO
        score_home, _ = get_match_result(match['homeGoalCount'], match['awayGoalCount'])
        exp_h = expected_score(ratings[home], ratings[away], hfa=hfa)
        ratings[home] = update_rating(ratings[home], score_home, exp_h, k)
        ratings[away] = update_rating(ratings[away], 1.0 - score_home, 1.0 - exp_h, k)

    records_df = pd.DataFrame(records)

    # Definir faixas
    bins = [0, 50, 100, 150, 200, 300, float('inf')]
    labels = ['0-50', '50-100', '100-150', '150-200', '200-300', '300+']
    records_df['faixa'] = pd.cut(records_df['elo_diff'], bins=bins, labels=labels, right=False)

    draw_analysis = records_df.groupby('faixa', observed=False).agg(
        n_jogos=('is_draw', 'count'),
        empates=('is_draw', 'sum'),
    ).reset_index()
    draw_analysis['draw_rate'] = draw_analysis['empates'] / draw_analysis['n_jogos']

    # Taxa global
    global_rate = records_df['is_draw'].mean()

    return draw_analysis, global_rate


def export_ratings(ratings, history_df, filepath=None):
    """Exporta ratings atuais para CSV."""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "elo_ratings.csv")

    last_match = history_df.groupby('team').agg(
        last_match_unix=('date_unix', 'max'),
        last_division=('division', 'last'),
    ).reset_index()

    df = pd.DataFrame([
        {'team': team, 'elo_rating': round(elo, 1)}
        for team, elo in ratings.items()
    ])
    df = df.merge(last_match, on='team', how='left')
    df = df.sort_values('elo_rating', ascending=False).reset_index(drop=True)
    df.to_csv(filepath, index=False)
    return df


def plot_hfa_calibration(results_df):
    """Gráfico de Brier Score vs HFA."""
    import plotly.graph_objects as go

    best_idx = results_df['brier_score'].idxmin()
    best_hfa = results_df.loc[best_idx, 'hfa']
    best_brier = results_df.loc[best_idx, 'brier_score']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results_df['hfa'],
        y=results_df['brier_score'],
        mode='lines+markers',
        name='Brier Score',
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=[best_hfa],
        y=[best_brier],
        mode='markers+text',
        name=f'Ótimo: HFA={best_hfa}',
        marker=dict(size=12, color='red', symbol='star'),
        text=[f'HFA={best_hfa}<br>BS={best_brier:.5f}'],
        textposition='top center',
    ))
    fig.update_layout(
        title="Calibração de Home Field Advantage — Grid Search",
        xaxis_title="HFA (pontos ELO adicionados ao mandante)",
        yaxis_title="Brier Score (menor = melhor)",
        template='plotly_white',
    )
    return fig


def plot_draw_rate(draw_analysis_df, global_rate):
    """Gráfico de taxa de empates por faixa de diferença de ELO."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=draw_analysis_df['faixa'].astype(str),
        y=draw_analysis_df['draw_rate'],
        text=[f"{r:.1%}<br>(n={n})" for r, n in zip(draw_analysis_df['draw_rate'], draw_analysis_df['n_jogos'])],
        textposition='outside',
        name='Taxa de empates',
    ))
    fig.add_hline(
        y=global_rate,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Média global: {global_rate:.1%}",
    )
    fig.update_layout(
        title="Taxa de Empates por Diferença de ELO",
        xaxis_title="|ELO(mandante) + HFA - ELO(visitante)|",
        yaxis_title="Taxa de empates",
        yaxis_tickformat='.0%',
        template='plotly_white',
    )
    return fig


def plot_elo_evolution(history_df, teams=None):
    """Gera gráfico de evolução de ELO com Plotly."""
    import plotly.graph_objects as go

    if teams is None:
        last_elos = history_df.groupby('team')['elo_after'].last().sort_values()
        teams = list(last_elos.tail(2).index) + list(last_elos.head(2).index)

    filtered = history_df[history_df['team'].isin(teams)].copy()
    filtered['date'] = pd.to_datetime(filtered['date_unix'], unit='s')

    fig = go.Figure()
    for team in teams:
        team_data = filtered[filtered['team'] == team].sort_values('date')
        fig.add_trace(go.Scatter(
            x=team_data['date'],
            y=team_data['elo_after'],
            mode='lines',
            name=team,
            hovertemplate=(
                '%{x|%Y-%m-%d}<br>ELO: %{y:.0f}'
                '<extra>%{fullData.name}</extra>'
            ),
        ))

    fig.update_layout(
        title="Evolução de ELO — Séries A/B/C (2021-2026)",
        xaxis_title="Data",
        yaxis_title="Rating ELO",
        hovermode='x unified',
        template='plotly_white',
    )
    return fig


if __name__ == "__main__":
    print(f"Janela ELO: {ELO_WINDOW_START}-{TARGET_YEAR}")
    print(f"K progressivo por ano: {K_PROGRESSIVE}")
    print(f"Peso por divisão: {K_BASE_DIVISION}")
    print(f"Rating inicial: {RATING_INICIAL}")
    print()

    print("Carregando partidas (A+B+C)...")
    matches = load_historical_matches()
    by_div = matches.groupby('division').size()
    total = len(matches)
    print(f"  {total} partidas total")
    for div in ['A', 'B', 'C']:
        n = by_div.get(div, 0)
        print(f"    Série {div}: {n} ({100*n/total:.0f}%)")

    print("\nCalculando ELOs...")
    ratings, history = calculate_all_elos(matches)

    print("Exportando ratings...")
    os.makedirs(DATA_DIR, exist_ok=True)
    df = export_ratings(ratings, history)

    # Mostrar ranking separado por última divisão
    for div_label in ['A', 'B', 'C']:
        sub = df[df['last_division'] == div_label]
        if len(sub) == 0:
            continue
        print(f"\n=== ELO — times atualmente na Série {div_label} (top 10) ===")
        print(sub.head(10)[['team', 'elo_rating']].to_string(index=False))

    print(f"\n=== Bottom 5 geral ===")
    print(df.tail(5)[['team', 'elo_rating', 'last_division']].to_string(index=False))

    spread = df['elo_rating'].max() - df['elo_rating'].min()
    print(f"\nSpread 1º-último: {spread:.0f} pontos")
    print(f"Total de times: {len(df)}")

    print("\nGerando gráfico...")
    # 2 top Série A + 1 promovido recente + 1 rebaixado recente
    top2_a = df[df['last_division'] == 'A'].head(2)['team'].tolist()
    bot1_a = df[df['last_division'] == 'A'].tail(1)['team'].tolist()
    top1_b = df[df['last_division'] == 'B'].head(1)['team'].tolist()
    fig = plot_elo_evolution(history, teams=top2_a + bot1_a + top1_b)
    fig.write_html(os.path.join(DATA_DIR, "elo_evolution.html"))
    print("Gráfico salvo em data/elo_evolution.html")
