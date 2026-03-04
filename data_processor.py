"""BRForecast — Processador de dados para o dashboard Streamlit.

Carrega dados dos CSVs e do pipeline (ELO, Poisson, Match Analysis),
aplica cache do Streamlit, e transforma em formatos prontos para Plotly.

Quando o banco SQLite nao esta disponivel (ex: Streamlit Cloud),
todas as funcoes usam CSVs pre-exportados como fallback.
"""

import json
import pandas as pd
import numpy as np
import os
import sys

# Ensure src package is importable
sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TARGET_YEAR = 2026

# Detect if the SQLite database is available (local dev vs Cloud)
try:
    from src.config import DB_PATH
    _DB_AVAILABLE = os.path.exists(DB_PATH)
except Exception:
    _DB_AVAILABLE = False


def _cache_data(ttl=3600, **kwargs):
    """Decorator lazy: aplica st.cache_data somente quando o Streamlit runtime esta ativo.

    Na importacao (sem runtime), retorna a funcao sem cache.
    Na execucao via 'streamlit run', aplica cache transparentemente na primeira chamada.
    """
    def decorator(func):
        _wrapped = [None]  # mutable container for lazy init

        def wrapper(*args, **kw):
            if _wrapped[0] is None:
                try:
                    import streamlit.runtime.scriptrunner as _sr
                    _sr.get_script_run_ctx()
                    import streamlit as st
                    _wrapped[0] = st.cache_data(ttl=ttl, **kwargs)(func)
                except Exception:
                    _wrapped[0] = func
            return _wrapped[0](*args, **kw)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# =========================================================================
# 0. CSV fallback helpers (for when DB is unavailable)
# =========================================================================

@_cache_data(ttl=3600)
def _load_completed_csv(year=TARGET_YEAR):
    """Carrega jogos realizados do CSV pre-exportado."""
    path = os.path.join(DATA_DIR, f"completed_matches_{year}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ['team_a_xg', 'team_b_xg']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _load_league_avgs():
    """Carrega league_avgs do JSON pre-exportado."""
    path = os.path.join(DATA_DIR, "league_avgs.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        'avg_home_goals': 1.5, 'avg_away_goals': 1.2,
        'avg_total': 2.7, 'metric': 'xG', 'n_matches': 0,
    }


# =========================================================================
# 1. CSV Loaders (fast — always available)
# =========================================================================

@_cache_data(ttl=3600)
def load_simulation_results():
    """Carrega resultados agregados da simulacao Monte Carlo."""
    path = os.path.join(DATA_DIR, "simulation_results.csv")
    return pd.read_csv(path)


@_cache_data(ttl=3600)
def load_team_stats(year=TARGET_YEAR):
    """Carrega stats unificados: classificacao + ELO + forcas Poisson."""
    path = os.path.join(DATA_DIR, f"team_stats_{year}.csv")
    return pd.read_csv(path)


@_cache_data(ttl=3600)
def load_remaining_matches(year=TARGET_YEAR):
    """Carrega jogos restantes da temporada."""
    path = os.path.join(DATA_DIR, f"jogos_restantes_{year}.csv")
    return pd.read_csv(path)


@_cache_data(ttl=3600)
def load_current_table(year=TARGET_YEAR):
    """Carrega classificacao atual."""
    path = os.path.join(DATA_DIR, f"tabela_atual_{year}.csv")
    return pd.read_csv(path)


@_cache_data(ttl=3600)
def load_elo_csv():
    """Carrega ratings ELO do CSV."""
    path = os.path.join(DATA_DIR, "elo_ratings.csv")
    return pd.read_csv(path)


@_cache_data(ttl=86400)
def load_team_logos():
    """Carrega URLs de logos dos times a partir do banco FootyStats.

    Returns:
        dict {team_name: full_logo_url}
    """
    import sqlite3
    from src.config import DB_PATH, SERIE_A_IDS

    CDN_BASE = "https://cdn.footystats.org/img/"
    logos = {}
    try:
        conn = sqlite3.connect(DB_PATH)
        # Get logos from matches (uses short team names like 'Flamengo')
        for sid in SERIE_A_IDS.values():
            rows = conn.execute(
                "SELECT DISTINCT home_name, home_image FROM matches "
                "WHERE competition_id = ?", (sid,),
            ).fetchall()
            for name, img in rows:
                if name and img and name not in logos:
                    url = img if img.startswith("http") else CDN_BASE + img
                    logos[name] = url
        conn.close()
    except Exception:
        pass
    return logos


# =========================================================================
# 2. Heavy computations (lazy, cached)
# =========================================================================

@_cache_data(ttl=3600, show_spinner="Calculando ELOs...")
def compute_elo_data():
    """Calcula ELO ratings e historico completo.

    Returns:
        elo_ratings: dict {team: rating}
        elo_history: DataFrame com historico completo
    """
    if _DB_AVAILABLE:
        from src.elo import load_historical_matches, calculate_all_elos
        matches = load_historical_matches()
        ratings, history = calculate_all_elos(matches)
        return ratings, history

    # Fallback: load from CSVs
    elo_csv = load_elo_csv()
    ratings = dict(zip(elo_csv['team'], elo_csv['elo_rating']))
    history_path = os.path.join(DATA_DIR, "elo_history.csv")
    if os.path.exists(history_path):
        history = pd.read_csv(history_path)
    else:
        history = pd.DataFrame()
    return ratings, history


@_cache_data(ttl=3600, show_spinner="Calculando forcas dos times...")
def compute_strengths(year=TARGET_YEAR):
    """Calcula forcas ofensivas/defensivas e medias da liga.

    Returns:
        team_strengths: DataFrame
        league_avgs: dict
    """
    if _DB_AVAILABLE:
        from src.config import MIN_MATCHES_SEASON
        from src.poisson import load_serie_a_with_xg, calculate_team_strengths

        matches_xg = load_serie_a_with_xg()
        target = matches_xg[matches_xg['season_year'] == year]
        target_done = target[target['status'] == 'complete']

        if len(target_done) < MIN_MATCHES_SEASON:
            expanded = matches_xg[matches_xg['season_year'] >= year - 1]
            return calculate_team_strengths(expanded)
        return calculate_team_strengths(target_done)

    # Fallback: load from CSVs
    path = os.path.join(DATA_DIR, f"team_strengths_{year}.csv")
    if os.path.exists(path):
        team_strengths = pd.read_csv(path)
    else:
        team_strengths = pd.DataFrame(columns=['team', 'attack', 'defense', 'matches', 'avg_for', 'avg_against'])
    league_avgs = _load_league_avgs()
    return team_strengths, league_avgs


@_cache_data(ttl=3600)
def compute_xpts(year=TARGET_YEAR):
    """Calcula xPTS acumulado por time via Poisson a partir do xG de cada jogo.

    Para cada jogo, usa o xG de cada time como lambda na distribuicao de Poisson,
    gera a matriz de probabilidades de placar, e calcula:
        xPTS = 3 * P(win) + 1 * P(draw)

    Ref: https://mckayjohns.substack.com/p/how-to-calculate-expected-points
    """
    from scipy.stats import poisson

    if _DB_AVAILABLE:
        from src.config import SERIE_A_IDS
        from src.load_data import load_completed_matches
        season_id = SERIE_A_IDS[year]
        completed = load_completed_matches(season_id)
    else:
        completed = _load_completed_csv(year)

    if len(completed) == 0:
        return pd.DataFrame(columns=["team", "xpts"])

    max_goals = 7
    goal_range = np.arange(max_goals + 1)

    xpts = {}
    for _, m in completed.iterrows():
        home = m["home_name"]
        away = m["away_name"]
        xg_h = m.get("team_a_xg", np.nan)
        xg_a = m.get("team_b_xg", np.nan)

        if pd.isna(xg_h) or pd.isna(xg_a):
            continue

        xpts.setdefault(home, 0.0)
        xpts.setdefault(away, 0.0)

        # Poisson probability matrix
        h_probs = poisson.pmf(goal_range, max(xg_h, 0.01))
        a_probs = poisson.pmf(goal_range, max(xg_a, 0.01))
        matrix = np.outer(h_probs, a_probs)

        p_home_win = np.sum(np.tril(matrix, -1))
        p_draw = np.sum(np.diag(matrix))
        p_away_win = np.sum(np.triu(matrix, 1))

        xpts[home] += 3.0 * p_home_win + 1.0 * p_draw
        xpts[away] += 3.0 * p_away_win + 1.0 * p_draw

    return pd.DataFrame(
        [{"team": t, "xpts": round(p, 1)} for t, p in xpts.items()]
    )


@_cache_data(ttl=3600)
def compute_match_breakdown(year=TARGET_YEAR):
    """Retorna detalhamento por jogo para cada time (para tooltips).

    Returns:
        dict[team] -> list of dicts com:
            opp, loc(C/F), gf, ga, xgf, xga, xpts_match
    """
    from scipy.stats import poisson

    if _DB_AVAILABLE:
        from src.config import SERIE_A_IDS
        from src.load_data import load_completed_matches
        season_id = SERIE_A_IDS[year]
        completed = load_completed_matches(season_id)
    else:
        completed = _load_completed_csv(year)

    if len(completed) == 0:
        return {}

    max_goals = 7
    goal_range = np.arange(max_goals + 1)

    breakdown = {}
    for _, m in completed.iterrows():
        home = m["home_name"]
        away = m["away_name"]
        xg_h = m.get("team_a_xg", np.nan)
        xg_a = m.get("team_b_xg", np.nan)

        if pd.isna(xg_h) or pd.isna(xg_a):
            continue

        # Poisson xPTS per match
        h_probs = poisson.pmf(goal_range, max(xg_h, 0.01))
        a_probs = poisson.pmf(goal_range, max(xg_a, 0.01))
        matrix = np.outer(h_probs, a_probs)
        p_home_win = np.sum(np.tril(matrix, -1))
        p_draw = np.sum(np.diag(matrix))
        p_away_win = np.sum(np.triu(matrix, 1))

        xpts_home = 3.0 * p_home_win + 1.0 * p_draw
        xpts_away = 3.0 * p_away_win + 1.0 * p_draw

        breakdown.setdefault(home, []).append({
            "opp": away, "loc": "C",
            "gf": int(m["homeGoalCount"]), "ga": int(m["awayGoalCount"]),
            "xgf": xg_h, "xga": xg_a, "xpts": xpts_home,
        })
        breakdown.setdefault(away, []).append({
            "opp": home, "loc": "F",
            "gf": int(m["awayGoalCount"]), "ga": int(m["homeGoalCount"]),
            "xgf": xg_a, "xga": xg_h, "xpts": xpts_away,
        })

    return breakdown


# =========================================================================
# 4. ELO History (para pagina de Evolucao ELO)
# =========================================================================

@_cache_data(ttl=3600, show_spinner="Calculando historico ELO...")
def load_elo_history():
    """Calcula ELO completo e retorna (ratings_dict, history_df).

    history_df colunas: date_unix, season_year, division, game_week,
    team, elo_before, elo_after, opponent, is_home, goals_for,
    goals_against, result, k_used.
    """
    if _DB_AVAILABLE:
        from src.elo import load_historical_matches, calculate_all_elos
        matches = load_historical_matches()
        ratings, history_df = calculate_all_elos(matches)
        history_df["date"] = pd.to_datetime(history_df["date_unix"], unit="s")
        return ratings, history_df

    # Fallback: load from CSVs
    elo_csv = load_elo_csv()
    ratings = dict(zip(elo_csv['team'], elo_csv['elo_rating']))
    history_path = os.path.join(DATA_DIR, "elo_history.csv")
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        if 'date' not in history_df.columns and 'date_unix' in history_df.columns:
            history_df["date"] = pd.to_datetime(history_df["date_unix"], unit="s")
    else:
        history_df = pd.DataFrame()
    return ratings, history_df


@_cache_data(ttl=3600, show_spinner="Analisando jogos realizados...")
def compute_match_cards(year=TARGET_YEAR):
    """Gera fichas de analise para todos os jogos realizados.

    Returns:
        DataFrame com uma linha por jogo e metricas de merecimento.
    """
    from src.match_analysis import analyze_all_matches

    if _DB_AVAILABLE:
        from src.config import SERIE_A_IDS
        from src.load_data import load_completed_matches
        season_id = SERIE_A_IDS[year]
        completed = load_completed_matches(season_id)
    else:
        completed = _load_completed_csv(year)

    if len(completed) == 0:
        return pd.DataFrame()

    elo_ratings, _ = compute_elo_data()
    team_strengths, league_avgs = compute_strengths(year)

    cards = analyze_all_matches(
        completed, team_strengths, league_avgs, elo_ratings=elo_ratings,
    )
    return cards


# =========================================================================
# 3. What-if simulation (not cached — dynamic input)
# =========================================================================

def run_whatif(year, fixed_results, n_sims=5000):
    """Roda Monte Carlo com resultados fixados.

    Args:
        year: ano da temporada
        fixed_results: dict {(mandante, visitante): (gols_h, gols_a)}
        n_sims: numero de simulacoes

    Returns:
        DataFrame agregado (mesma estrutura de simulation_results.csv)
    """
    from src.simulation import run_monte_carlo_whatif, aggregate_results

    table = load_current_table(year)
    remaining = load_remaining_matches(year)
    elo_ratings, _ = compute_elo_data()
    team_strengths, league_avgs = compute_strengths(year)

    positions, points, team_names = run_monte_carlo_whatif(
        n_sims, remaining, table, team_strengths, league_avgs,
        elo_ratings=elo_ratings, fixed_results=fixed_results,
        show_progress=False,
    )

    return aggregate_results(positions, points, team_names, n_sims)


# =========================================================================
# 4. Probability calculations for upcoming matches
# =========================================================================

@_cache_data(ttl=3600)
def compute_upcoming_probs(year=TARGET_YEAR):
    """Calcula probabilidades H/D/A para cada jogo restante.

    Returns:
        DataFrame [rodada, mandante, visitante, p_home, p_draw, p_away]
    """
    from src.poisson import calculate_lambdas, score_probabilities

    remaining = load_remaining_matches(year)
    elo_ratings, _ = compute_elo_data()
    team_strengths, league_avgs = compute_strengths(year)

    rows = []
    for _, match in remaining.iterrows():
        home = match['mandante']
        away = match['visitante']

        lam_h, lam_a = calculate_lambdas(
            home, away, team_strengths, league_avgs,
            elo_ratings=elo_ratings,
        )
        probs = score_probabilities(lam_h, lam_a)

        rows.append({
            'rodada': match['rodada'],
            'mandante': home,
            'visitante': away,
            'p_home': probs['home_win'],
            'p_draw': probs['draw'],
            'p_away': probs['away_win'],
            'lambda_home': lam_h,
            'lambda_away': lam_a,
        })

    return pd.DataFrame(rows)


# =========================================================================
# 5. Formatting helpers
# =========================================================================

ZONE_COLORS = {
    'libertadores': '#2ECC71',
    'sulamericana': '#3498DB',
    'neutro': '#95A5A6',
    'rebaixamento': '#E74C3C',
}


def get_zone(position):
    """Retorna a zona baseada na posicao."""
    if position <= 6:
        return 'libertadores'
    elif position <= 12:
        return 'sulamericana'
    elif position <= 16:
        return 'neutro'
    return 'rebaixamento'


def get_zone_color(position):
    """Retorna cor da zona baseada na posicao."""
    return ZONE_COLORS[get_zone(position)]


def get_zone_label(position):
    """Retorna label da zona."""
    z = get_zone(position)
    labels = {
        'libertadores': 'Libertadores',
        'sulamericana': 'Sul-Americana',
        'neutro': '',
        'rebaixamento': 'Rebaixamento',
    }
    return labels[z]


def format_pct(val):
    """Formata probabilidade como string percentual."""
    if pd.isna(val) or val == 0:
        return "—"
    if val >= 0.01:
        return f"{val:.1%}"
    return f"{val:.2%}"


def result_emoji(result):
    """Retorna indicador visual para resultado."""
    return {'H': 'V', 'D': 'E', 'A': 'D'}.get(result, '?')


def verdict_color(verdict):
    """Retorna cor para veredicto de merecimento."""
    return {
        'Merecido': '#2ECC71',
        'Parcialmente injusto': '#F39C12',
        'Muito injusto': '#E74C3C',
        'Sem xG': '#95A5A6',
    }.get(verdict, '#95A5A6')
