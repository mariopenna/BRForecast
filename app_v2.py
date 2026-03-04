"""BRForecast v2 — Dashboard Streamlit.

Dashboard interativo para previsao do Campeonato Brasileiro Serie A.

Uso:
    streamlit run app_v2.py
"""

import html as html_lib
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(__file__))

from data_processor import (
    load_current_table, load_team_stats, compute_xpts,
    compute_match_breakdown, load_elo_history, load_simulation_results,
    compute_upcoming_probs, compute_match_cards, get_zone_color,
    verdict_color, load_team_logos, ZONE_COLORS, TARGET_YEAR,
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
# Sidebar — Navegacao
# =========================================================================

st.sidebar.markdown(
    '<h1 style="color:#ffffff !important;margin:0;padding:0">BRForecast</h1>',
    unsafe_allow_html=True,
)
st.sidebar.caption(f"Campeonato Brasileiro Serie A {TARGET_YEAR}")

SECTIONS = [
    "Classificacao Atual",
    "Evolucao ELO",
    "Previsao Final",
    "Deep Dive por Time",
    "Visao Individual dos Jogos",
    "Sobre o Projeto",
]

section = st.sidebar.radio("Navegacao", SECTIONS)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dados:** FootyStats | "
    "**Modelo:** ELO + Poisson/Dixon-Coles + Monte Carlo"
)

# =========================================================================
# Custom CSS
# =========================================================================

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .cover-title {
        text-align: center; font-size: 3rem; font-weight: 800;
        margin-bottom: 0; color: #1a1a2e;
    }
    .cover-subtitle {
        text-align: center; font-size: 1.3rem; color: #666;
        margin-top: 0; margin-bottom: 2rem;
    }
    .cover-divider {
        border: none; height: 3px;
        background: linear-gradient(90deg, #2ECC71, #3498DB, #E74C3C);
        margin: 0 auto 2rem auto; width: 60%; border-radius: 2px;
    }
    /* Sidebar title branco */
    [data-testid="stSidebar"] h1 {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# Capa
# =========================================================================

st.markdown('<h1 class="cover-title">BRForecast</h1>', unsafe_allow_html=True)
st.markdown(
    f'<p class="cover-subtitle">Campeonato Brasileiro Serie A — {TARGET_YEAR}</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="cover-divider">', unsafe_allow_html=True)


# =========================================================================
# Helper: logo inline HTML
# =========================================================================

def _logo_html(logos_dict, team, size=18):
    """Retorna tag <img> do logo do time ou string vazia."""
    url = logos_dict.get(team, "")
    if url:
        return (
            f'<img src="{url}" width="{size}" height="{size}" '
            f'style="vertical-align:middle;margin-right:5px">'
        )
    return ""


# =========================================================================
# SECTION: Classificacao Atual
# =========================================================================

def section_classificacao():
    st.subheader("Classificacao Atual")

    table = load_current_table()
    stats = load_team_stats()
    xpts_df = compute_xpts()
    breakdown = compute_match_breakdown()

    # Merge
    table = table.merge(
        stats[["team", "avg_xg_for", "avg_xg_against"]],
        on="team", how="left",
    )
    table = table.merge(
        xpts_df[["team", "xpts"]],
        on="team", how="left",
    )
    table = table.sort_values("posicao").reset_index(drop=True)

    # Dados para exibicao
    display = table[[
        "posicao", "team", "pontos", "jogos",
        "vitorias", "empates", "derrotas",
        "gols_pro", "gols_contra",
        "xpts",
    ]].copy()

    display["xg_total"] = (table["avg_xg_for"] * table["jogos"]).round(1)
    display["xga_total"] = (table["avg_xg_against"] * table["jogos"]).round(1)
    display["gp_diff"] = (display["gols_pro"] - display["xg_total"]).round(1)
    display["gc_diff"] = (display["gols_contra"] - display["xga_total"]).round(1)
    display["pts_diff"] = (display["pontos"] - display["xpts"]).round(1)

    # --- Tooltip helpers ---
    def _esc(text):
        return html_lib.escape(str(text), quote=True)

    def _tt_gp(team):
        matches = breakdown.get(team, [])
        if not matches:
            return ""
        lines = [f"{'C' if m['loc']=='C' else 'F'} vs {m['opp']}: {m['gf']} gol(s)" for m in matches]
        return _esc("\n".join(lines))

    def _tt_gc(team):
        matches = breakdown.get(team, [])
        if not matches:
            return ""
        lines = [f"{'C' if m['loc']=='C' else 'F'} vs {m['opp']}: {m['ga']} gol(s)" for m in matches]
        return _esc("\n".join(lines))

    def _tt_xg(team):
        matches = breakdown.get(team, [])
        if not matches:
            return ""
        lines = [f"{'C' if m['loc']=='C' else 'F'} vs {m['opp']}: {m['xgf']:.2f} xG" for m in matches]
        return _esc("\n".join(lines))

    def _tt_xga(team):
        matches = breakdown.get(team, [])
        if not matches:
            return ""
        lines = [f"{'C' if m['loc']=='C' else 'F'} vs {m['opp']}: {m['xga']:.2f} xGA" for m in matches]
        return _esc("\n".join(lines))

    def _tt_pts(team):
        matches = breakdown.get(team, [])
        if not matches:
            return ""
        lines = []
        for m in matches:
            res = "V" if m["gf"] > m["ga"] else ("E" if m["gf"] == m["ga"] else "D")
            pts = 3 if res == "V" else (1 if res == "E" else 0)
            lines.append(
                f"{'C' if m['loc']=='C' else 'F'} vs {m['opp']}: "
                f"{m['gf']}x{m['ga']} ({res}) +{pts}pt"
            )
        return _esc("\n".join(lines))

    def _tt_xpts(team):
        matches = breakdown.get(team, [])
        if not matches:
            return ""
        lines = [f"{'C' if m['loc']=='C' else 'F'} vs {m['opp']}: {m['xpts']:.2f} xPTS" for m in matches]
        return _esc("\n".join(lines))

    def _tt_ved(team, result_filter):
        matches = breakdown.get(team, [])
        if not matches:
            return ""
        lines = []
        for m in matches:
            res = "V" if m["gf"] > m["ga"] else ("E" if m["gf"] == m["ga"] else "D")
            if res == result_filter:
                lines.append(f"{'C' if m['loc']=='C' else 'F'} vs {m['opp']}: {m['gf']}x{m['ga']}")
        return _esc("\n".join(lines) if lines else "Nenhum")

    # --- Style helpers ---
    def _zone_bg(pos):
        if pos <= 6:
            return "rgba(46,204,113,0.12)"
        elif pos <= 12:
            return "rgba(52,152,219,0.10)"
        elif pos <= 16:
            return "transparent"
        return "rgba(231,76,60,0.12)"

    def _diff_color(val, invert=False):
        if pd.isna(val) or val == 0:
            return "#333"
        if invert:
            return "#1e7e34" if val < 0 else "#c62828"
        return "#1e7e34" if val > 0 else "#c62828"

    def _diff_fmt(val):
        if pd.isna(val):
            return "-"
        return f"+{val:.1f}" if val > 0 else f"{val:.1f}"

    # --- Build HTML rows ---
    logos = load_team_logos()
    rows_html = []
    for _, r in display.iterrows():
        team = r["team"]
        pos = int(r["posicao"])
        bg = _zone_bg(pos)
        c = []

        c.append(f'<td class="n">{pos}</td>')
        c.append(f'<td class="tm">{_logo_html(logos, team)}{_esc(team)}</td>')
        c.append(f'<td class="n">{int(r["jogos"])}</td>')
        c.append(f'<td class="n" title="{_tt_ved(team,"V")}">{int(r["vitorias"])}</td>')
        c.append(f'<td class="n" title="{_tt_ved(team,"E")}">{int(r["empates"])}</td>')
        c.append(f'<td class="n" title="{_tt_ved(team,"D")}">{int(r["derrotas"])}</td>')

        c.append(f'<td class="n sec-gp" title="{_tt_gp(team)}">{int(r["gols_pro"])}</td>')
        c.append(f'<td class="n" title="{_tt_xg(team)}">{r["xg_total"]:.1f}</td>')
        d = r["gp_diff"]
        c.append(f'<td class="n" style="color:{_diff_color(d)};font-weight:600">{_diff_fmt(d)}</td>')

        c.append(f'<td class="n sec-gc" title="{_tt_gc(team)}">{int(r["gols_contra"])}</td>')
        c.append(f'<td class="n" title="{_tt_xga(team)}">{r["xga_total"]:.1f}</td>')
        d = r["gc_diff"]
        c.append(f'<td class="n" style="color:{_diff_color(d, invert=True)};font-weight:600">{_diff_fmt(d)}</td>')

        c.append(f'<td class="n sec-pt" title="{_tt_pts(team)}">{int(r["pontos"])}</td>')
        c.append(f'<td class="n" title="{_tt_xpts(team)}">{r["xpts"]:.1f}</td>')
        d = r["pts_diff"]
        c.append(f'<td class="n" style="color:{_diff_color(d)};font-weight:600">{_diff_fmt(d)}</td>')

        rows_html.append(f'<tr style="background:{bg}">{"".join(c)}</tr>')

    # --- Full HTML table ---
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
      font-size:13px;background:#fff}}
table{{width:100%;border-collapse:collapse}}
.gr th{{padding:8px 4px;font-size:.72rem;font-weight:700;
        text-transform:uppercase;letter-spacing:.05em;text-align:center}}
.g-pf{{background:#f0f1f3;color:#333}}
.g-gp{{background:#d4edda;color:#1e7e34;border-left:2px solid #c3e6cb}}
.g-gc{{background:#f8d7da;color:#c62828;border-left:2px solid #f5c6cb}}
.g-pt{{background:#cce5ff;color:#1565c0;border-left:2px solid #b8daff}}
.cr th{{padding:8px 6px;font-weight:600;font-size:.78rem;text-align:center;
        background:#fafafa;border-bottom:2px solid #ddd;
        cursor:pointer;user-select:none}}
.cr th:hover{{background:#eee}}
.cr th .arr{{font-size:.55rem;margin-left:2px;color:#bbb}}
.cr th.asc .arr::after{{content:" ▲";color:#333}}
.cr th.desc .arr::after{{content:" ▼";color:#333}}
.cr .s-gp{{border-left:2px solid #c3e6cb}}
.cr .s-gc{{border-left:2px solid #f5c6cb}}
.cr .s-pt{{border-left:2px solid #b8daff}}
td{{padding:7px 6px;border-bottom:1px solid #eee;text-align:center;white-space:nowrap}}
td.tm{{text-align:left;font-weight:500;padding-left:10px}}
td.n{{font-variant-numeric:tabular-nums}}
.sec-gp{{border-left:2px solid #c3e6cb}}
.sec-gc{{border-left:2px solid #f5c6cb}}
.sec-pt{{border-left:2px solid #b8daff}}
tbody tr:hover{{filter:brightness(0.95)}}
td[title]{{cursor:help;text-decoration:underline dotted #ccc;text-underline-offset:3px}}
td.tm{{text-decoration:none;cursor:default}}
</style></head><body>
<table id="tbl">
<thead>
  <tr class="gr">
    <th colspan="6" class="g-pf">Performance</th>
    <th colspan="3" class="g-gp">Gols Pro</th>
    <th colspan="3" class="g-gc">Gols Contra</th>
    <th colspan="3" class="g-pt">Pontos</th>
  </tr>
  <tr class="cr">
    <th data-c="0" data-t="n">#<span class="arr"></span></th>
    <th data-c="1" data-t="t">Time<span class="arr"></span></th>
    <th data-c="2" data-t="n">J<span class="arr"></span></th>
    <th data-c="3" data-t="n">V<span class="arr"></span></th>
    <th data-c="4" data-t="n">E<span class="arr"></span></th>
    <th data-c="5" data-t="n">D<span class="arr"></span></th>
    <th data-c="6" data-t="n" class="s-gp">GP<span class="arr"></span></th>
    <th data-c="7" data-t="n">xG<span class="arr"></span></th>
    <th data-c="8" data-t="n">xG+/-<span class="arr"></span></th>
    <th data-c="9" data-t="n" class="s-gc">GC<span class="arr"></span></th>
    <th data-c="10" data-t="n">xGA<span class="arr"></span></th>
    <th data-c="11" data-t="n">xGA+/-<span class="arr"></span></th>
    <th data-c="12" data-t="n" class="s-pt">P<span class="arr"></span></th>
    <th data-c="13" data-t="n">xPTS<span class="arr"></span></th>
    <th data-c="14" data-t="n">P+/-<span class="arr"></span></th>
  </tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>
<script>
(function(){{
  const tbl=document.getElementById('tbl');
  const ths=tbl.querySelectorAll('.cr th');
  let cur={{c:-1,asc:true}};
  ths.forEach(th=>{{
    th.addEventListener('click',function(){{
      const c=+this.dataset.c, t=this.dataset.t;
      const asc=(cur.c===c)?!cur.asc:(t==='t');
      ths.forEach(h=>h.classList.remove('asc','desc'));
      this.classList.add(asc?'asc':'desc');
      cur={{c,asc}};
      const tb=tbl.querySelector('tbody');
      const rows=Array.from(tb.rows);
      rows.sort((a,b)=>{{
        let va=a.cells[c].textContent.trim();
        let vb=b.cells[c].textContent.trim();
        if(t==='n'){{
          va=parseFloat(va.replace('+',''))||0;
          vb=parseFloat(vb.replace('+',''))||0;
          return asc?va-vb:vb-va;
        }}
        return asc?va.localeCompare(vb):vb.localeCompare(va);
      }});
      rows.forEach(r=>tb.appendChild(r));
    }});
  }});
}})();
</script>
</body></html>"""

    components.html(full_html, height=810, scrolling=False)

    # Legenda das zonas
    st.markdown("")
    leg1, leg2, leg3, leg4 = st.columns(4)
    leg1.markdown("🟢 **1-6** Libertadores")
    leg2.markdown("🔵 **7-12** Sul-Americana")
    leg3.markdown("⚪ **13-16** Meio de tabela")
    leg4.markdown("🔴 **17-20** Rebaixamento")

    # Glossario
    st.markdown("")
    st.caption("**Glossario**")
    st.markdown("""
| Coluna | Descricao |
|--------|-----------|
| **J / V / E / D** | Jogos, Vitorias, Empates, Derrotas |
| **GP** | Gols marcados |
| **xG** | Expected Goals |
| **xG+/-** | Saldo de xG vs Gols marcados |
| **GC** | Gols sofridos |
| **xGA** | Expected Goals Against |
| **xGA+/-** | Saldo de xG vs Gols sofridos |
| **P** | Pontos reais acumulados |
| **xPTS** | Pontos esperados |
| **P+/-** | Saldo de Pontos vs Esperados |
""")


# =========================================================================
# SECTION: Evolucao ELO
# =========================================================================

def section_elo():
    st.subheader("Evolucao ELO")

    ratings, elo_history = load_elo_history()

    # Times da Serie A e B atuais (2026), ordenados por ELO mais recente (desc)
    def _sort_by_latest_elo(teams_series):
        teams = teams_series.unique()
        latest_elo = {}
        for t in teams:
            td = elo_history[elo_history["team"] == t].sort_values("date")
            if not td.empty:
                latest_elo[t] = td["elo_after"].iloc[-1]
            else:
                latest_elo[t] = 0
        return sorted(teams, key=lambda t: latest_elo[t], reverse=True)

    serie_a_teams = _sort_by_latest_elo(
        elo_history[
            (elo_history["season_year"] == TARGET_YEAR)
            & (elo_history["division"] == "A")
        ]["team"]
    )
    serie_b_teams = _sort_by_latest_elo(
        elo_history[
            (elo_history["season_year"] == TARGET_YEAR)
            & (elo_history["division"] == "B")
        ]["team"]
    )
    all_teams = serie_a_teams + serie_b_teams

    # Historico 2025 + 2026 para todos esses times
    elo_2y = elo_history[
        (elo_history["team"].isin(all_teams))
        & (elo_history["season_year"] >= 2025)
    ].copy().sort_values("date")

    # Cores representativas de cada time
    TEAM_COLORS = {
        # Verdes
        "Palmeiras": "#006633",
        "Coritiba": "#4CAF50",
        "Chapecoense": "#00897B",
        "Goiás": "#2E7D32",
        "América MG": "#388E3C",
        "Juventude": "#1B5E20",
        "Guarani": "#43A047",
        # Azuis
        "Cruzeiro": "#1565C0",
        "Grêmio": "#42A5F5",
        "Bahia": "#0068B4",
        "Remo": "#00BCD4",
        "Fortaleza": "#1976D2",
        "Avaí": "#0D47A1",
        "Paysandu": "#29B6F6",
        "CSA": "#1E88E5",
        # Vermelhos
        "Flamengo": "#D32F2F",
        "Internacional": "#FF1744",
        "Atlético PR": "#8B0000",
        "Vitória": "#FF5722",
        "São Paulo": "#E91E63",
        "Sport": "#C62828",
        "CRB": "#EF5350",
        "Vila Nova": "#B71C1C",
        # Laranja / Amarelo
        "Bragantino": "#FF6D00",
        "Mirassol": "#FFD600",
        "Novorizontino": "#F9A825",
        # Escuros / Neutros
        "Corinthians": "#1A1A1A",
        "Atlético Mineiro": "#555555",
        "Botafogo": "#C9A900",
        "Vasco da Gama": "#37474F",
        "Santos": "#757575",
        "Ceará": "#2C2C2C",
        "Ponte Preta": "#424242",
        "Operário PR": "#3E3E3E",
        # Vinho / Roxo
        "Fluminense": "#880E4F",
    }

    # Detectar gap entre temporadas para comprimir o eixo X
    dates_sorted = pd.Series(elo_2y["date"].unique()).sort_values().reset_index(drop=True)
    xaxis_rangebreaks = []
    season_break_date = None
    if len(dates_sorted) > 1:
        diffs = dates_sorted.diff().iloc[1:]
        max_gap_idx = diffs.dt.days.idxmax()
        gap_start = dates_sorted[max_gap_idx - 1]
        gap_end = dates_sorted[max_gap_idx]
        gap_days = (gap_end - gap_start).days
        if gap_days > 30:
            rb_start = gap_start + pd.Timedelta(days=3)
            rb_end = gap_end - pd.Timedelta(days=3)
            xaxis_rangebreaks = [dict(bounds=[rb_start, rb_end])]
            season_break_date = gap_end

    # --- Grafico Plotly com legenda clicavel a direita ---
    fig = go.Figure()

    # Serie A — visivel por default
    for team in serie_a_teams:
        td = elo_2y[elo_2y["team"] == team].sort_values("date")
        if td.empty:
            continue
        fig.add_trace(go.Scatter(
            x=td["date"],
            y=td["elo_after"],
            mode="lines",
            name=f"{team}  (A)",
            line=dict(width=2, color=TEAM_COLORS.get(team)),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "%{x|%d/%m/%Y}<br>"
                "ELO: %{y:.0f}"
                "<extra></extra>"
            ),
        ))

    # Serie B — oculto por default (legendonly)
    for team in serie_b_teams:
        td = elo_2y[elo_2y["team"] == team].sort_values("date")
        if td.empty:
            continue
        fig.add_trace(go.Scatter(
            x=td["date"],
            y=td["elo_after"],
            mode="lines",
            name=f"{team}  (B)",
            line=dict(width=1.5, dash="dot", color=TEAM_COLORS.get(team)),
            visible="legendonly",
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "%{x|%d/%m/%Y}<br>"
                "ELO: %{y:.0f}"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(
        y=1500, line_dash="dot", line_color="rgba(150,150,150,0.5)",
        annotation_text="Media (1500)",
        annotation_position="bottom right",
    )

    # Linha vertical na transicao de temporada
    if season_break_date is not None:
        x_str = season_break_date.strftime("%Y-%m-%d")
        fig.add_vline(
            x=x_str, line_dash="dash",
            line_color="rgba(100,100,100,0.4)", line_width=1,
        )
        fig.add_annotation(
            x=x_str, y=1, yref="paper",
            text="2026", showarrow=False,
            font=dict(size=10, color="#888"),
            xanchor="left", yanchor="top",
        )

    fig.update_layout(
        title="Evolucao ELO — Series A & B (2025-2026)",
        xaxis_title="",
        xaxis=dict(rangebreaks=xaxis_rangebreaks),
        yaxis_title="Rating ELO",
        hovermode="x unified",
        template="plotly_white",
        height=650,
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(r=180),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Clique no nome do time na legenda para ocultar/mostrar. "
        "Duplo-clique para isolar um time."
    )

    # --- Tabela de tendencia: ultimos 5 jogos (apenas 2026) ---
    st.markdown("")
    st.subheader("Tendencia — Ultimos 5 jogos")

    elo_2026 = elo_history[
        (elo_history["season_year"] == TARGET_YEAR)
    ].sort_values("date")

    def _esc(text):
        return html_lib.escape(str(text), quote=True)

    def _dcol(val):
        if val == 0:
            return "#888"
        return "#1e7e34" if val > 0 else "#c62828"

    def _dfmt(val):
        return f"+{val:.1f}" if val > 0 else f"{val:.1f}"

    trend_rows_html = []
    # Collect data for sorting
    trend_data = []
    for team in serie_a_teams + serie_b_teams:
        td = elo_2026[elo_2026["team"] == team].sort_values("date")
        if td.empty:
            continue
        last5 = td.tail(5).iloc[::-1]  # mais recente primeiro
        deltas = (last5["elo_after"].values - last5["elo_before"].values).round(1)
        current_elo = round(ratings.get(team, 0), 1)
        total_var = round(sum(deltas), 1)
        trend_data.append((team, current_elo, total_var, deltas, last5))

    # Ordenar por ELO desc (default)
    trend_data.sort(key=lambda x: x[1], reverse=True)

    for team, current_elo, total_var, deltas, last5 in trend_data:
        cells = []

        # Time com tooltip ELO + var
        team_tip = _esc(f"ELO: {current_elo:.0f} ({_dfmt(total_var)} nos ultimos 5J)")
        cells.append(f'<td class="tm" title="{team_tip}">{_esc(team)}</td>')
        cells.append(f'<td class="n">{current_elo:.0f}</td>')

        # Var 5J com tooltip resumo dos 5 jogos
        rows_list = last5.to_dict("records")
        var_lines = []
        for i in range(min(len(deltas), 5)):
            m = rows_list[i]
            loc = "C" if m["is_home"] else "F"
            gf, ga = int(m["goals_for"]), int(m["goals_against"])
            var_lines.append(
                f"{m['opponent']} ({loc}) {gf}x{ga}: {_dfmt(deltas[i])}"
            )
        var_tip = _esc("\n".join(var_lines)) if var_lines else ""
        cells.append(
            f'<td class="n" title="{var_tip}" '
            f'style="color:{_dcol(total_var)};font-weight:700">'
            f'{_dfmt(total_var)}</td>'
        )

        # J-1 a J-5 com tooltip do adversario
        for i in range(5):
            if i < len(deltas):
                d = deltas[i]
                m = rows_list[i]
                loc = "C" if m["is_home"] else "F"
                res = m["result"]
                gf = int(m["goals_for"])
                ga = int(m["goals_against"])
                tip = _esc(
                    f"vs {m['opponent']} ({loc})\n"
                    f"{gf}x{ga} ({res})\n"
                    f"ELO: {m['elo_before']:.0f} → {m['elo_after']:.0f}"
                )
                cells.append(
                    f'<td class="n" title="{tip}" '
                    f'style="color:{_dcol(d)};font-weight:600">'
                    f'{_dfmt(d)}</td>'
                )
            else:
                cells.append('<td class="n">-</td>')

        trend_rows_html.append(f'<tr>{"".join(cells)}</tr>')

    trend_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
      font-size:13px;background:#fff}}
table{{width:100%;border-collapse:collapse}}
thead th{{padding:8px 6px;font-weight:600;font-size:.78rem;text-align:center;
          background:#fafafa;border-bottom:2px solid #ddd;
          cursor:pointer;user-select:none}}
thead th:hover{{background:#eee}}
thead th .arr{{font-size:.55rem;margin-left:2px;color:#bbb}}
thead th.asc .arr::after{{content:" ▲";color:#333}}
thead th.desc .arr::after{{content:" ▼";color:#333}}
td{{padding:7px 6px;border-bottom:1px solid #eee;text-align:center;white-space:nowrap}}
td.tm{{text-align:left;font-weight:500;padding-left:10px}}
td.n{{font-variant-numeric:tabular-nums}}
tbody tr:hover{{background:rgba(0,0,0,0.03)}}
td[title]{{cursor:help;text-decoration:underline dotted #ccc;text-underline-offset:3px}}
</style></head><body>
<table id="trend">
<thead><tr>
  <th data-c="0" data-t="t">Time<span class="arr"></span></th>
  <th data-c="1" data-t="n">ELO<span class="arr"></span></th>
  <th data-c="2" data-t="n">Var 5J<span class="arr"></span></th>
  <th data-c="3" data-t="n">J-1<span class="arr"></span></th>
  <th data-c="4" data-t="n">J-2<span class="arr"></span></th>
  <th data-c="5" data-t="n">J-3<span class="arr"></span></th>
  <th data-c="6" data-t="n">J-4<span class="arr"></span></th>
  <th data-c="7" data-t="n">J-5<span class="arr"></span></th>
</tr></thead>
<tbody>
{"".join(trend_rows_html)}
</tbody>
</table>
<script>
(function(){{
  const tbl=document.getElementById('trend');
  const ths=tbl.querySelectorAll('thead th');
  let cur={{c:-1,asc:true}};
  ths.forEach(th=>{{
    th.addEventListener('click',function(){{
      const c=+this.dataset.c,t=this.dataset.t;
      const asc=(cur.c===c)?!cur.asc:(t==='t');
      ths.forEach(h=>h.classList.remove('asc','desc'));
      this.classList.add(asc?'asc':'desc');
      cur={{c,asc}};
      const tb=tbl.querySelector('tbody');
      const rows=Array.from(tb.rows);
      rows.sort((a,b)=>{{
        let va=a.cells[c].textContent.trim();
        let vb=b.cells[c].textContent.trim();
        if(t==='n'){{
          va=parseFloat(va.replace('+',''))||0;
          vb=parseFloat(vb.replace('+',''))||0;
          return asc?va-vb:vb-va;
        }}
        return asc?va.localeCompare(vb):vb.localeCompare(va);
      }});
      rows.forEach(r=>tb.appendChild(r));
    }});
  }});
}})();
</script>
</body></html>"""

    n_teams = len(trend_data)
    components.html(trend_html, height=min(50 + n_teams * 36, 800), scrolling=True)

    st.caption(
        "Passe o mouse sobre J-1 a J-5 para ver adversario, placar e variacao de ELO. "
        "Clique nos headers para ordenar."
    )

    # Glossario ELO
    st.markdown("")
    st.caption("**Glossario**")
    st.markdown("""
| Termo | Descricao |
|-------|-----------|
| **ELO** | Rating de forca do time |
| **Var 5J** | Soma da variacao de ELO nos ultimos 5 jogos de 2026. Indica tendencia de forma recente |
| **J-1 a J-5** | Variacao de ELO em cada jogo (J-1 = mais recente). Verde = ganhou ELO, vermelho = perdeu |
| **Regressao** | No inicio de cada temporada, os ratings regridem parcialmente a 1500 para evitar inflacao |
""")


# =========================================================================
# SECTION: Previsao Final
# =========================================================================

def section_previsao():
    st.subheader("Previsao Final")

    sim = load_simulation_results()
    table = load_current_table()

    # Merge pontos atuais
    sim = sim.merge(
        table[["team", "pontos", "jogos"]],
        on="team", how="left",
    )
    sim = sim.sort_values("pts_mean", ascending=False).reset_index(drop=True)

    # --- Tabela Expected Points (barras horizontais via HTML) ---
    st.markdown("")
    st.subheader("Expected Points")
    st.caption("Pontos esperados ao final do campeonato (media das simulacoes Monte Carlo)")

    def _esc(text):
        return html_lib.escape(str(text), quote=True)

    logos = load_team_logos()
    max_pts = sim["pts_mean"].max()

    xp_rows = []
    for _, r in sim.iterrows():
        team = r["team"]
        pts_mean = r["pts_mean"]
        pts_now = int(r["pontos"]) if pd.notna(r["pontos"]) else 0
        pts_p10 = r["pts_p10"]
        pts_p90 = r["pts_p90"]
        bar_w = (pts_mean / max_pts * 100) if max_pts > 0 else 0

        # Gradient color based on position (green top -> pale bottom)
        rank = _ + 1
        n = len(sim)
        t = rank / n  # 0 = top, 1 = bottom
        # Green gradient: dark green -> light green -> pale yellow
        r_c = int(30 + t * 200)
        g_c = int(140 + t * 80)
        b_c = int(50 + t * 100)
        bar_color = f"rgb({r_c},{g_c},{b_c})"

        tip = _esc(
            f"{team}\n"
            f"Pts atuais: {pts_now}\n"
            f"Pts esperados: {pts_mean:.1f}"
        )

        xp_rows.append(
            f'<tr title="{tip}">'
            f'<td class="tm">{_logo_html(logos, team)}{_esc(team)}</td>'
            f'<td class="bar-cell">'
            f'<div class="bar-bg">'
            f'<div class="bar" style="width:{bar_w:.1f}%;background:{bar_color}"></div>'
            f'</div>'
            f'</td>'
            f'<td class="n">{pts_mean:.1f}</td>'
            f'</tr>'
        )

    xp_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
      font-size:13px;background:#fff}}
table{{width:100%;border-collapse:collapse}}
thead th{{padding:8px 6px;font-weight:600;font-size:.78rem;text-align:center;
          background:#fafafa;border-bottom:2px solid #ddd}}
td{{padding:6px 6px;border-bottom:1px solid #eee;white-space:nowrap}}
td.tm{{text-align:left;font-weight:500;padding-left:10px;width:140px}}
td.n{{text-align:center;font-variant-numeric:tabular-nums;width:55px}}
td.bar-cell{{padding:6px 8px;width:auto}}
.bar-bg{{background:#f0f0f0;border-radius:3px;height:20px;width:100%}}
.bar{{height:100%;border-radius:3px;min-width:2px}}
tbody tr:hover{{background:rgba(0,0,0,0.03)}}
tr[title]{{cursor:help}}
</style></head><body>
<table>
<thead><tr>
  <th style="text-align:left;padding-left:10px">Time</th>
  <th>Expected Points</th>
  <th>Pts Esp.</th>
</tr></thead>
<tbody>
{"".join(xp_rows)}
</tbody>
</table>
</body></html>"""

    n_teams = len(sim)
    components.html(xp_html, height=min(50 + n_teams * 32, 750), scrolling=True)

    st.caption("Passe o mouse sobre uma linha para ver detalhes.")

    # --- Heatmap de Posicoes Finais ---
    st.markdown("")
    st.subheader("Heatmap de Posicoes Finais")
    st.caption("Probabilidade de cada time terminar em cada posicao (Monte Carlo)")

    sim_sorted = sim.sort_values("pos_mean")
    pos_cols = [f"pos_{i}" for i in range(1, 21)]
    teams = sim_sorted["team"].tolist()

    matrix = np.zeros((len(teams), 20))
    for i, team in enumerate(teams):
        row = sim_sorted[sim_sorted["team"] == team].iloc[0]
        for j in range(20):
            matrix[i, j] = row[pos_cols[j]]

    # Text annotations: show % only when >= 1%
    # 1 digito (< 10%) -> mostra com 1 decimal (ex: 8,3%)
    # 2+ digitos (>= 10%) -> sem decimal (ex: 39%)
    text_matrix = []
    for i in range(len(teams)):
        row_text = []
        for j in range(20):
            v = matrix[i, j]
            if v >= 0.10:
                row_text.append(f"{v:.0%}")
            elif v >= 0.01:
                row_text.append(f"{v:.1%}".replace(".", ","))
            else:
                row_text.append("")
        text_matrix.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[str(p) for p in range(1, 21)],
        y=teams,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#000"),
        colorscale=[
            [0.0, "#FFFFFF"],
            [0.05, "#FFF9C4"],
            [0.15, "#FFE082"],
            [0.30, "#FF8A65"],
            [0.50, "#E53935"],
            [1.0, "#B71C1C"],
        ],
        hovertemplate=(
            "%{y}<br>"
            "Posicao %{x}: %{z:.1%}"
            "<extra></extra>"
        ),
        colorbar=dict(title="Prob.", tickformat=".0%"),
    ))

    # Linhas divisorias de zonas (eixo X categorico: indice 0 = "1", ..., 19 = "20")
    for idx in [5.5, 11.5, 15.5]:  # entre pos 6-7, 12-13, 16-17
        fig.add_vline(
            x=idx, line_dash="dash", line_color="black",
            line_width=1, opacity=0.5,
        )

    # Anotacoes de zona (centradas em cada faixa)
    fig.add_annotation(
        x=2.5, y=-0.8, text="Libertadores",
        showarrow=False, font=dict(color=ZONE_COLORS["libertadores"], size=11),
    )
    fig.add_annotation(
        x=8.5, y=-0.8, text="Sul-Americana",
        showarrow=False, font=dict(color=ZONE_COLORS["sulamericana"], size=11),
    )
    fig.add_annotation(
        x=17.5, y=-0.8, text="Rebaixamento",
        showarrow=False, font=dict(color=ZONE_COLORS["rebaixamento"], size=11),
    )

    fig.update_layout(
        xaxis_title="Posicao Final",
        yaxis=dict(autorange="reversed"),
        height=700,
        template="plotly_white",
        margin=dict(l=150, b=80),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Glossario
    st.markdown("")
    st.caption("**Glossario**")
    st.markdown("""
| Termo | Descricao |
|-------|-----------|
| **Pts Esp.** | Media de pontos ao final do campeonato em 20.000 simulacoes Monte Carlo |
| **Heatmap** | Cores mais quentes = maior probabilidade de terminar naquela posicao |
| **Libertadores** | Posicoes 1-6: classificacao para a Copa Libertadores |
| **Sul-Americana** | Posicoes 7-12: classificacao para a Copa Sul-Americana |
| **Rebaixamento** | Posicoes 17-20: rebaixamento para a Serie B |
""")


# =========================================================================
# SECTION: Deep Dive por Time
# =========================================================================

def section_deep_dive():
    st.subheader("Deep Dive por Time")

    sim = load_simulation_results()
    stats = load_team_stats()
    table = load_current_table()

    # Merge pontos atuais
    sim = sim.merge(
        table[["team", "pontos", "jogos", "posicao"]],
        on="team", how="left",
    )

    teams_sorted = sim.sort_values("pos_mean")["team"].tolist()
    selected = st.selectbox("Selecione o time", teams_sorted)

    team_sim = sim[sim["team"] == selected].iloc[0]
    team_stats = stats[stats["team"] == selected]
    team_stats = team_stats.iloc[0] if len(team_stats) > 0 else None

    # --- Metricas resumo ---
    col1, col2, col3, col4, col5 = st.columns(5)
    pos_atual = team_sim.get("posicao", None)
    col1.metric("Posicao Atual", f"{pos_atual:.0f}" if pd.notna(pos_atual) else "-")
    col2.metric("Titulo", f"{team_sim['p_titulo']:.1%}")
    col3.metric("Libertadores", f"{team_sim['p_libertadores']:.1%}")
    col4.metric("Sul-Americana", f"{team_sim['p_sulamericana']:.1%}")
    col5.metric("Rebaixamento", f"{team_sim['p_rebaixamento']:.1%}")

    # --- Stats do time ---
    if team_stats is not None:
        st.markdown("---")
        scol1, scol2, scol3, scol4, scol5 = st.columns(5)
        scol1.metric("Jogos", f"{team_stats['jogos']:.0f}")
        scol2.metric("Pontos", f"{team_stats['pontos']:.0f}")
        scol3.metric("Pts Esperados", f"{team_sim['pts_mean']:.1f}")
        scol4.metric("Ataque", f"{team_stats['attack']:.2f}")
        scol5.metric("Defesa", f"{team_stats['defense']:.2f}")

    st.markdown("---")

    # --- Distribuicao de posicoes ---
    st.subheader("Distribuicao de Posicoes Finais")
    pos_cols = [f"pos_{i}" for i in range(1, 21)]
    pos_probs = [team_sim[col] for col in pos_cols]
    colors = [get_zone_color(i + 1) for i in range(20)]

    fig_pos = go.Figure(go.Bar(
        x=list(range(1, 21)),
        y=pos_probs,
        marker_color=colors,
        hovertemplate="Posicao %{x}: %{y:.1%}<extra></extra>",
    ))
    fig_pos.update_layout(
        xaxis_title="Posicao",
        yaxis_title="Probabilidade",
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=350,
        xaxis=dict(dtick=1),
    )
    st.plotly_chart(fig_pos, use_container_width=True)

    # --- Evolucao de ELO na temporada ---
    st.subheader("Evolucao de ELO na Temporada")
    _, elo_history = load_elo_history()

    team_hist = elo_history[
        (elo_history["team"] == selected)
        & (elo_history["season_year"] >= TARGET_YEAR - 1)
    ].copy().sort_values("date")

    if len(team_hist) > 0:
        # Build hover info with match details
        team_hist["loc"] = team_hist["is_home"].apply(lambda h: "C" if h else "F")
        team_hist["gf"] = team_hist["goals_for"].astype(int)
        team_hist["ga"] = team_hist["goals_against"].astype(int)
        team_hist["hover_text"] = team_hist.apply(
            lambda r: (
                f"{r['date'].strftime('%d/%m/%Y')}<br>"
                f"vs {r['opponent']} ({r['loc']}) {r['gf']}x{r['ga']}<br>"
                f"ELO: {r['elo_before']:.0f} → {r['elo_after']:.0f}"
            ), axis=1,
        )

        # Detectar gap entre temporadas para comprimir eixo X
        dd_dates = pd.Series(
            team_hist["date"].unique(),
        ).sort_values().reset_index(drop=True)
        dd_rangebreaks = []
        dd_break_date = None
        if len(dd_dates) > 1:
            dd_diffs = dd_dates.diff().iloc[1:]
            dd_max_idx = dd_diffs.dt.days.idxmax()
            dd_gap_start = dd_dates[dd_max_idx - 1]
            dd_gap_end = dd_dates[dd_max_idx]
            dd_gap_days = (dd_gap_end - dd_gap_start).days
            if dd_gap_days > 30:
                dd_rb_start = dd_gap_start + pd.Timedelta(days=3)
                dd_rb_end = dd_gap_end - pd.Timedelta(days=3)
                dd_rangebreaks = [dict(bounds=[dd_rb_start, dd_rb_end])]
                dd_break_date = dd_gap_end

        fig_elo = go.Figure()
        fig_elo.add_trace(go.Scatter(
            x=team_hist["date"],
            y=team_hist["elo_after"],
            mode="lines",
            name=selected,
            line=dict(width=2),
            hoverinfo="skip",
        ))

        # Pontos coloridos por resultado (V/E/D)
        result_colors = {"W": "#2ECC71", "D": "#F39C12", "L": "#E74C3C"}
        result_labels = {"W": "Vitoria", "D": "Empate", "L": "Derrota"}
        for result, color in result_colors.items():
            subset = team_hist[team_hist["result"] == result]
            if subset.empty:
                continue
            fig_elo.add_trace(go.Scatter(
                x=subset["date"],
                y=subset["elo_after"],
                mode="markers",
                name=result_labels[result],
                marker=dict(size=8, color=color),
                text=subset["hover_text"],
                hovertemplate="%{text}<extra></extra>",
            ))

        # Linha vertical na transicao de temporada
        if dd_break_date is not None:
            dd_x_str = dd_break_date.strftime("%Y-%m-%d")
            fig_elo.add_vline(
                x=dd_x_str, line_dash="dash",
                line_color="rgba(100,100,100,0.4)", line_width=1,
            )
            fig_elo.add_annotation(
                x=dd_x_str, y=1, yref="paper",
                text=str(TARGET_YEAR), showarrow=False,
                font=dict(size=10, color="#888"),
                xanchor="left", yanchor="top",
            )

        fig_elo.update_layout(
            xaxis_title="Data",
            xaxis=dict(rangebreaks=dd_rangebreaks),
            yaxis_title="Rating ELO",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig_elo, use_container_width=True)
    else:
        st.info("Sem historico ELO para a temporada atual.")

    # --- Proximos jogos ---
    st.subheader("Proximos Jogos")
    upcoming = compute_upcoming_probs()
    team_matches = upcoming[
        (upcoming["mandante"] == selected) | (upcoming["visitante"] == selected)
    ].head(10)

    if len(team_matches) > 0:
        rows = []
        for _, m in team_matches.iterrows():
            is_home = m["mandante"] == selected
            opponent = m["visitante"] if is_home else m["mandante"]
            mando = "Casa" if is_home else "Fora"
            p_win = m["p_home"] if is_home else m["p_away"]
            p_lose = m["p_away"] if is_home else m["p_home"]

            rows.append({
                "Rodada": f"{m['rodada']:.0f}",
                "Adversario": opponent,
                "Mando": mando,
                f"Previsão de Vitória {selected}": f"{p_win:.1%}",
                "Previsão de Empate": f"{m['p_draw']:.1%}",
                f"Previsão de Derrota {selected}": f"{p_lose:.1%}",
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("Nenhum jogo restante.")

    # Glossario
    st.markdown("")
    st.caption("**Glossario**")
    st.markdown("""
| Termo | Descricao |
|-------|-----------|
| **Posicao Atual** | Posicao na tabela com base nos jogos ja realizados |
| **Titulo** | Probabilidade de terminar em 1o lugar nas simulacoes Monte Carlo |
| **Libertadores** | Probabilidade de terminar entre 1o-6o (vaga na Libertadores) |
| **Sul-Americana** | Probabilidade de terminar entre 7o-12o (vaga na Sul-Americana) |
| **Rebaixamento** | Probabilidade de terminar entre 17o-20o (rebaixado para Serie B) |
| **Pts Esperados** | Media de pontos ao final do campeonato (Monte Carlo) |
| **ELO** | Rating de forca do time. Pontos verdes = vitoria, amarelos = empate, vermelhos = derrota |
""")


# =========================================================================
# SECTION: Visao Individual dos Jogos
# =========================================================================

def section_jogos():
    st.subheader("Visao Individual dos Jogos")

    cards = compute_match_cards()
    if cards.empty:
        st.info("Nenhum jogo realizado ainda.")
        return

    # Sort by game_week descending (most recent first)
    cards = cards.sort_values(
        ["game_week", "date_unix"], ascending=[False, False],
    ).reset_index(drop=True)

    # --- Filters ---
    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])

    rodadas = sorted(cards["game_week"].dropna().unique(), reverse=True)
    with col_f1:
        rodada_sel = st.selectbox(
            "Rodada", ["Todas"] + [f"Rodada {int(r)}" for r in rodadas],
        )

    teams_all = sorted(
        set(cards["home"].tolist() + cards["away"].tolist())
    )
    with col_f2:
        team_filter = st.selectbox("Time", ["Todos"] + teams_all)

    verdicts_all = ["Merecido", "Parcialmente injusto", "Muito injusto", "Sem xG"]
    with col_f3:
        verdict_filter = st.selectbox("Merecimento", ["Todos"] + verdicts_all)

    # Apply filters
    filtered = cards.copy()
    if rodada_sel != "Todas":
        r_num = int(rodada_sel.replace("Rodada ", ""))
        filtered = filtered[filtered["game_week"] == r_num]
    if team_filter != "Todos":
        filtered = filtered[
            (filtered["home"] == team_filter) | (filtered["away"] == team_filter)
        ]
    if verdict_filter != "Todos":
        filtered = filtered[filtered["verdict"] == verdict_filter]

    if filtered.empty:
        st.info("Nenhum jogo encontrado com esses filtros.")
        return

    # --- Match list table (HTML) ---
    def _esc(t):
        return html_lib.escape(str(t), quote=True)

    verdict_colors = {
        "Merecido": "#2ECC71",
        "Parcialmente injusto": "#F39C12",
        "Muito injusto": "#E74C3C",
        "Sem xG": "#95A5A6",
    }

    logos = load_team_logos()
    trows = []
    for _, r in filtered.iterrows():
        gw = int(r["game_week"]) if pd.notna(r["game_week"]) else ""
        home = _esc(r["home"])
        away = _esc(r["away"])
        gh, ga = int(r["goals_home"]), int(r["goals_away"])
        placar = f"{gh} x {ga}"

        if r["has_xg"]:
            xg_placar = f"{r['xg_home']:.1f} x {r['xg_away']:.1f}"
        else:
            xg_placar = "—"

        v = r["verdict"]
        vc = verdict_colors.get(v, "#95A5A6")
        badge = (
            f'<span style="background:{vc};color:#fff;padding:2px 8px;'
            f'border-radius:10px;font-size:.75rem;font-weight:500">'
            f'{_esc(v)}</span>'
        )

        date_str = _esc(r["date"]) if r["date"] else ""

        trows.append(
            f'<tr>'
            f'<td class="n">{gw}</td>'
            f'<td class="dt">{date_str}</td>'
            f'<td class="tm">{_logo_html(logos, r["home"], 16)}{home} vs {_logo_html(logos, r["away"], 16)}{away}</td>'
            f'<td class="n">{placar}</td>'
            f'<td class="n">{xg_placar}</td>'
            f'<td class="badge-cell">{badge}</td>'
            f'</tr>'
        )

    table_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
      font-size:13px;background:#fff}}
table{{width:100%;border-collapse:collapse}}
thead th{{padding:8px 6px;font-weight:600;font-size:.78rem;text-align:center;
          background:#fafafa;border-bottom:2px solid #ddd;position:sticky;top:0;z-index:1}}
td{{padding:7px 6px;border-bottom:1px solid #eee;white-space:nowrap}}
td.n{{text-align:center;font-variant-numeric:tabular-nums}}
td.dt{{text-align:center;color:#666;font-size:.8rem}}
td.tm{{text-align:left;font-weight:500;padding-left:10px}}
td.badge-cell{{text-align:center}}
tbody tr:hover{{background:rgba(0,0,0,0.04)}}
</style></head><body>
<table>
<thead><tr>
  <th>Rod.</th>
  <th>Data</th>
  <th style="text-align:left;padding-left:10px">Jogo</th>
  <th>Placar</th>
  <th>xG</th>
  <th>Merecimento</th>
</tr></thead>
<tbody>
{"".join(trows)}
</tbody>
</table>
</body></html>"""

    n_rows = len(filtered)
    components.html(table_html, height=min(50 + n_rows * 34, 600), scrolling=True)

    st.caption(f"{n_rows} jogos exibidos")

    # --- Detailed card for selected match ---
    st.markdown("---")
    st.subheader("Detalhe do Jogo")

    match_labels = []
    match_idx = []
    for _, r in filtered.iterrows():
        gw = int(r["game_week"]) if pd.notna(r["game_week"]) else "?"
        label = (
            f"R{gw} — {r['home']} {int(r['goals_home'])}x"
            f"{int(r['goals_away'])} {r['away']}"
        )
        match_labels.append(label)
        match_idx.append(_)

    selected_label = st.selectbox(
        "Selecione o jogo", match_labels,
        index=None, placeholder="Escolha um jogo para ver detalhes...",
    )
    if selected_label is None:
        return

    sel_i = match_idx[match_labels.index(selected_label)]
    m = filtered.loc[sel_i]

    st.markdown("")

    # Header: Home  Score  Away
    hcol1, hcol2, hcol3 = st.columns([2, 1, 2])
    with hcol1:
        st.markdown(
            f'<p style="text-align:right;font-size:1.4rem;font-weight:600">'
            f'{m["home"]}</p>',
            unsafe_allow_html=True,
        )
    with hcol2:
        st.markdown(
            f'<p style="text-align:center;font-size:2rem;font-weight:700">'
            f'{int(m["goals_home"])} x {int(m["goals_away"])}</p>',
            unsafe_allow_html=True,
        )
    with hcol3:
        st.markdown(
            f'<p style="text-align:left;font-size:1.4rem;font-weight:600">'
            f'{m["away"]}</p>',
            unsafe_allow_html=True,
        )

    # Meta + verdict badge
    meta_parts = []
    if pd.notna(m["game_week"]):
        meta_parts.append(f"Rodada {int(m['game_week'])}")
    if m["date"]:
        meta_parts.append(m["date"])
    meta_parts.append(m["result_text"])
    meta_str = " | ".join(meta_parts)

    vc = verdict_color(m["verdict"])
    st.markdown(
        f'<p style="text-align:center;color:#666;margin-bottom:4px">{meta_str}</p>'
        f'<p style="text-align:center">'
        f'<span style="background:{vc};color:#fff;padding:4px 16px;'
        f'border-radius:12px;font-weight:600">{m["verdict"]}</span></p>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    if m["has_xg"]:
        # --- xG comparison chart ---
        st.markdown("**xG vs Gols**")
        fig_xg = go.Figure()
        fig_xg.add_trace(go.Bar(
            name="Gols",
            x=[m["home"], m["away"]],
            y=[int(m["goals_home"]), int(m["goals_away"])],
            marker_color=["#3498DB", "#E74C3C"],
            text=[int(m["goals_home"]), int(m["goals_away"])],
            textposition="outside",
        ))
        fig_xg.add_trace(go.Bar(
            name="xG",
            x=[m["home"], m["away"]],
            y=[m["xg_home"], m["xg_away"]],
            marker_color=["rgba(52,152,219,0.4)", "rgba(231,76,60,0.4)"],
            text=[f'{m["xg_home"]:.2f}', f'{m["xg_away"]:.2f}'],
            textposition="outside",
        ))
        fig_xg.update_layout(
            barmode="group",
            template="plotly_white",
            height=280,
            yaxis_title="",
            showlegend=True,
            legend=dict(orientation="h", y=1.12),
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_xg, use_container_width=True)

        # Luck info
    # --- Poisson model probabilities ---
    st.markdown("**Probabilidades do Modelo**")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    pcol1.metric(f"P(V {m['home']})", f"{m['p_home_win']:.1%}")
    pcol2.metric("P(Empate)", f"{m['p_draw']:.1%}")
    pcol3.metric(f"P(V {m['away']})", f"{m['p_away_win']:.1%}")
    pcol4.metric(
        f"P({int(m['goals_home'])}x{int(m['goals_away'])})",
        f"{m['p_exact_score']:.1%}",
    )

    # --- Odds comparison ---
    if m["has_odds"]:
        st.markdown("**Odds vs Modelo**")
        # raw values for bold detection
        modelo_vals = [m['p_home_win'], m['p_draw'], m['p_away_win']]
        odds_vals = [m['odds_home'], m['odds_draw'], m['odds_away']]
        impl_vals = [m['odds_p_home'], m['odds_p_draw'], m['odds_p_away']]
        max_mod = max(modelo_vals)
        min_odd = min(odds_vals)  # menor odd = favorito
        max_impl = max(impl_vals)
        odds_rows = [
            (f"V {m['home']}", m['p_home_win'], f"{m['p_home_win']:.1%}", m['odds_home'], f"{m['odds_home']:.2f}", m['odds_p_home'], f"{m['odds_p_home']:.1%}"),
            ("Empate", m['p_draw'], f"{m['p_draw']:.1%}", m['odds_draw'], f"{m['odds_draw']:.2f}", m['odds_p_draw'], f"{m['odds_p_draw']:.1%}"),
            (f"V {m['away']}", m['p_away_win'], f"{m['p_away_win']:.1%}", m['odds_away'], f"{m['odds_away']:.2f}", m['odds_p_away'], f"{m['odds_p_away']:.1%}"),
        ]
        b = "font-weight:700"
        odds_html_rows = ""
        for lbl, mv, mod, ov, odd, iv, imp in odds_rows:
            ms = f" style='{b}'" if mv == max_mod else ""
            os = f" style='{b}'" if ov == min_odd else ""
            ims = f" style='{b}'" if iv == max_impl else ""
            odds_html_rows += f"<tr><td class='lbl'>{lbl}</td><td{ms}>{mod}</td><td{ims}>{imp}</td><td{os}>{odd}</td></tr>"
        odds_table = f"""<table style="width:100%;border-collapse:collapse;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:14px">
<thead><tr>
  <th style="text-align:center;padding:8px;border-bottom:2px solid #ddd;width:25%"></th>
  <th style="text-align:center;padding:8px;border-bottom:2px solid #ddd;width:25%">Modelo</th>
  <th style="text-align:center;padding:8px;border-bottom:2px solid #ddd;width:25%">Impl. Odds</th>
  <th style="text-align:center;padding:8px;border-bottom:2px solid #ddd;width:25%">Odds</th>
</tr></thead>
<tbody>{odds_html_rows}</tbody>
</table>
<style>
table td {{padding:7px 6px;border-bottom:1px solid #eee;text-align:center;font-variant-numeric:tabular-nums}}
table td.lbl {{font-weight:500;color:#333}}
</style>"""
        st.markdown(odds_table, unsafe_allow_html=True)

    # --- Extra stats ---
    st.markdown("**Estatisticas**")
    # (label, home_key, away_key, format)  format: "int" or "float"
    stat_defs = [
        ("Placar", "goals_home", "goals_away", "int"),
        ("xG", "xg_home", "xg_away", "float"),
        ("Chutes", "shots_home", "shots_away", "int"),
        ("Chutes no Alvo", "shots_on_target_home", "shots_on_target_away", "int"),
        ("Escanteios", "corners_home", "corners_away", "int"),
        ("Ataques Perigosos", "dangerous_attacks_home", "dangerous_attacks_away", "int"),
        ("Faltas", "fouls_home", "fouls_away", "int"),
    ]
    stat_html_rows = []
    for label, h_col, a_col, fmt in stat_defs:
        vh = m.get(h_col)
        va = m.get(a_col)
        if vh is not None and va is not None and pd.notna(vh) and pd.notna(va):
            if fmt == "float":
                vh_f, va_f = float(vh), float(va)
                vh_str, va_str = f"{vh_f:.2f}", f"{va_f:.2f}"
                h_style = " style='font-weight:700'" if vh_f > va_f else ""
                a_style = " style='font-weight:700'" if va_f > vh_f else ""
            else:
                vh_int, va_int = int(vh), int(va)
                vh_str, va_str = str(vh_int), str(va_int)
                h_style = " style='font-weight:700'" if vh_int > va_int else ""
                a_style = " style='font-weight:700'" if va_int > vh_int else ""
            stat_html_rows.append(
                f"<tr><td{h_style}>{vh_str}</td><td class='st'>{label}</td><td{a_style}>{va_str}</td></tr>"
            )
    if stat_html_rows:
        home_esc = html_lib.escape(str(m["home"]), quote=True)
        away_esc = html_lib.escape(str(m["away"]), quote=True)
        stat_table = f"""<table style="width:100%;border-collapse:collapse;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:14px">
<thead><tr>
  <th style="text-align:center;padding:8px;border-bottom:2px solid #ddd;width:25%">{home_esc}</th>
  <th style="text-align:center;padding:8px;border-bottom:2px solid #ddd;width:50%;color:#666;font-size:.85rem"></th>
  <th style="text-align:center;padding:8px;border-bottom:2px solid #ddd;width:25%">{away_esc}</th>
</tr></thead>
<tbody>{"".join(stat_html_rows)}</tbody>
</table>
<style>
table td {{padding:7px 6px;border-bottom:1px solid #eee;text-align:center;font-variant-numeric:tabular-nums}}
table td.st {{color:#666;font-size:.85rem;font-weight:500}}
</style>"""
        st.markdown(stat_table, unsafe_allow_html=True)

    # Glossario
    st.markdown("")
    st.caption("**Glossario**")
    st.markdown("""
| Termo | Descricao |
|-------|-----------|
| **Rod.** | Rodada do campeonato |
| **Placar** | Resultado final do jogo |
| **xG** | Expected Goals |
| **Merecido** | Resultado real alinhado com o xG  |
| **Parcialmente injusto** | Resultado diverge do xG, mas com desvio moderado |
| **Muito injusto** | Resultado muito diferente do que o xG indicava |
| **P(V) / P(E) / P(D)** | Probabilidades do modelo para vitoria, empate e derrota |
| **P(placar)** | Probabilidade do modelo para o placar exato que ocorreu |
| **Odds** | Cotacoes de mercado pre-jogo |
| **Impl. Odds** | Probabilidade implicita das odds |
""")


# =========================================================================
# SECTION: Sobre o Projeto
# =========================================================================

def section_sobre():
    st.subheader("Sobre o Projeto")

    st.markdown("""
O **BRForecast** e um modelo estatistico para previsao do Campeonato Brasileiro
Serie A. Combina tres tecnicas complementares para estimar probabilidades de
resultados, posicoes finais e zonas de classificacao.
""")

    # --- Pipeline ---
    st.markdown("---")
    st.markdown("### Como funciona")

    st.markdown("""
O pipeline do modelo segue quatro etapas:

**1. Rating ELO** — Mede a forca relativa de cada time com base no historico
de resultados. Times com ELO mais alto tendem a vencer mais. O rating e
atualizado jogo a jogo e incorpora Series A, B e C desde 2021.

**2. Modelo Poisson / Dixon-Coles** — Para cada confronto, estima o numero
esperado de gols (lambda) de cada time usando forcas ofensivas e defensivas
calculadas a partir do xG (Expected Goals). A correcao de Dixon-Coles ajusta
as probabilidades de placares baixos (0x0, 1x0, 0x1, 1x1), que o Poisson
puro tende a subestimar.

**3. Blend com Odds** — As probabilidades do modelo sao combinadas com as
odds de mercado (quando disponiveis). Isso aproveita a informacao do mercado
enquanto mantem a perspectiva do modelo.

**4. Simulacao Monte Carlo** — Simula os jogos restantes do campeonato 20.000
vezes. Em cada simulacao, o ELO e os lambdas sao atualizados a cada jogo
(hot update), capturando efeitos dinamicos como sequencias de vitorias ou
derrotas.
""")

    # --- Metricas ---
    st.markdown("---")
    st.markdown("### Glossario de Metricas")

    st.markdown("""
| Metrica | Descricao |
|---------|-----------|
| **ELO** | Sistema de rating onde cada time comeca em 1500 (Serie A). Ganha pontos ao vencer e perde ao ser derrotado. A magnitude depende do resultado esperado — vencer um favorito rende mais ELO |
| **HFA (Home Field Advantage)** | Bonus de ELO dado ao mandante para refletir a vantagem de jogar em casa. Calibrado em 105 pontos via grid search |
| **xG (Expected Goals)** | Gols esperados com base na qualidade e posicao das finalizacoes. Um xG de 1.5 significa que, em media, as chances criadas produziriam 1.5 gols |
| **Ataque** | Forca ofensiva relativa a media da liga. > 1.0 = ataque acima da media |
| **Defesa** | Forca defensiva relativa a media da liga. < 1.0 = defesa acima da media (sofre menos gols que o esperado) |
| **Dixon-Coles** | Correcao que ajusta a probabilidade de placares baixos (0x0, 1x0, etc.), que o Poisson puro tende a subestimar |
| **xPTS (Expected Points)** | Pontos esperados com base no xG. Calcula P(vitoria), P(empate) e P(derrota) via Poisson e converte em pontuacao esperada |
""")

    # --- Merecimento ---
    st.markdown("---")
    st.markdown("### Classificacao de Merecimento")

    st.markdown("""
Para cada jogo realizado, o modelo compara o xG com o placar real para
classificar o resultado:

| Veredicto | Criterio |
|-----------|----------|
| **Merecido** | O vencedor pelo xG e o mesmo vencedor pelo placar real (ou empate em ambos) |
| **Parcialmente injusto** | O vencedor pelo xG diverge do placar real, mas o desvio de sorte e moderado (≤ 1.5) |
| **Muito injusto** | O vencedor pelo xG diverge do placar real e o desvio de sorte e grande (> 1.5) |

**Sorte** = Gols - xG. Valor positivo significa que o time fez mais gols do que
as chances criadas justificavam.
""")

    # --- Zonas ---
    st.markdown("---")
    st.markdown("### Zonas do Campeonato")

    zcol1, zcol2, zcol3, zcol4 = st.columns(4)
    zcol1.markdown(
        f'<div style="background:{ZONE_COLORS["libertadores"]};color:#fff;'
        f'padding:12px;border-radius:8px;text-align:center">'
        f'<b>Libertadores</b><br>1a — 6a posicao</div>',
        unsafe_allow_html=True,
    )
    zcol2.markdown(
        f'<div style="background:{ZONE_COLORS["sulamericana"]};color:#fff;'
        f'padding:12px;border-radius:8px;text-align:center">'
        f'<b>Sul-Americana</b><br>7a — 12a posicao</div>',
        unsafe_allow_html=True,
    )
    zcol3.markdown(
        f'<div style="background:{ZONE_COLORS["neutro"]};color:#fff;'
        f'padding:12px;border-radius:8px;text-align:center">'
        f'<b>Zona neutra</b><br>13a — 16a posicao</div>',
        unsafe_allow_html=True,
    )
    zcol4.markdown(
        f'<div style="background:{ZONE_COLORS["rebaixamento"]};color:#fff;'
        f'padding:12px;border-radius:8px;text-align:center">'
        f'<b>Rebaixamento</b><br>17a — 20a posicao</div>',
        unsafe_allow_html=True,
    )

    # --- Fonte ---
    st.markdown("---")
    st.markdown("### Fonte de dados")

    st.markdown("""
- **Resultados, xG e estatisticas**: FootyStats (via scraping)
- **Odds de mercado**: FootyStats (media pre-jogo)
- **Cobertura**: Series A, B e C do Brasil (2021-2026 para ELO; Serie A foco para previsoes)
""")

    st.markdown("---")
    st.caption(
        "BRForecast — Modelo estatistico para fins educacionais. "
        "Nao constitui recomendacao de apostas."
    )


# =========================================================================
# Router
# =========================================================================

if section == "Classificacao Atual":
    section_classificacao()
elif section == "Evolucao ELO":
    section_elo()
elif section == "Previsao Final":
    section_previsao()
elif section == "Deep Dive por Time":
    section_deep_dive()
elif section == "Visao Individual dos Jogos":
    section_jogos()
elif section == "Sobre o Projeto":
    section_sobre()
