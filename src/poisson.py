"""BRForecast — Modelo Poisson + xG para previsao de placares.

Combina forcas ofensivas/defensivas calculadas via xG com ajuste por ELO.
- attack_strength = media xG gerado pelo time / media xG da liga
- defense_strength = media xGA sofrido pelo time / media xGA da liga
- lambda_home = attack_home x defense_away x avg_goals_home x elo_adj
- lambda_away = attack_away x defense_home x avg_goals_away x elo_adj

O ELO ajusta os lambdas via odds ratio elevado a ELO_LAMBDA_WEIGHT:
  elo_ratio = E_home / (1 - E_home)
  elo_adj_home = elo_ratio ^ weight
Quando weight=0 -> Poisson puro. Quanto maior, mais o ELO influencia.
"""

import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import poisson
import os

from src.config import (
    DB_PATH, DATA_DIR, SERIE_A_IDS, ELO_WINDOW_START, TARGET_YEAR,
    HFA, USE_XG, POISSON_MAX_GOALS, ELO_LAMBDA_WEIGHT,
    DRAW_BASE_RATE, MIN_MATCHES_SEASON, DIXON_COLES_RHO,
    BLEND_ALPHA, EMA_ALPHA,
)
from src.elo import expected_score


# ---------------------------------------------------------------------------
# 1. Carga de dados
# ---------------------------------------------------------------------------

def load_serie_a_with_xg(db_path=DB_PATH, start_year=ELO_WINDOW_START):
    """Carrega partidas da Serie A (2021+) com xG, gols e odds."""
    season_filter = {y: sid for y, sid in SERIE_A_IDS.items() if y >= start_year}
    season_ids = tuple(season_filter.values())

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"""
        SELECT m.home_name, m.away_name,
               m.homeID, m.awayID,
               m.homeGoalCount, m.awayGoalCount,
               m.team_a_xg, m.team_b_xg,
               m.odds_ft_1, m.odds_ft_x, m.odds_ft_2,
               m.date_unix, m.game_week, m.status,
               m.competition_id,
               l.year AS season_year
        FROM matches m
        JOIN leagues l ON m.competition_id = l.id
        WHERE m.competition_id IN {season_ids}
        ORDER BY m.date_unix
    """, conn)
    conn.close()

    # Converter xG para numerico
    for col in ['team_a_xg', 'team_b_xg']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ---------------------------------------------------------------------------
# 2. Forcas ofensivas e defensivas
# ---------------------------------------------------------------------------

def _compute_ema_strengths(valid_matches, col_home, col_away, alpha):
    """Calcula forcas via EMA (media movel exponencial).

    Processa partidas em ordem cronologica. Jogos recentes pesam mais:
        ema_t = alpha * valor_t + (1 - alpha) * ema_{t-1}

    Returns:
        DataFrame indexado por 'team' com colunas [avg_for, avg_against, matches]
    """
    sorted_m = valid_matches.sort_values('date_unix')
    ema_for = {}
    ema_against = {}
    counts = {}

    for _, match in sorted_m.iterrows():
        home = match['home_name']
        away = match['away_name']
        xg_h = float(match[col_home])
        xg_a = float(match[col_away])

        for team, xf, xa in [(home, xg_h, xg_a), (away, xg_a, xg_h)]:
            if team not in ema_for:
                ema_for[team] = xf
                ema_against[team] = xa
                counts[team] = 1
            else:
                ema_for[team] = alpha * xf + (1 - alpha) * ema_for[team]
                ema_against[team] = alpha * xa + (1 - alpha) * ema_against[team]
                counts[team] += 1

    rows = [{'team': t, 'avg_for': ema_for[t], 'avg_against': ema_against[t],
             'matches': counts[t]} for t in ema_for]
    return pd.DataFrame(rows).set_index('team')


def calculate_team_strengths(matches_df, use_xg=USE_XG, ema_alpha=EMA_ALPHA):
    """Calcula forca ofensiva e defensiva de cada time relativa a liga.

    attack  > 1.0 = ataca mais que a media (bom)
    defense > 1.0 = sofre  mais que a media (ruim)

    Quando ema_alpha > 0, usa media movel exponencial (jogos recentes pesam
    mais). Quando 0, usa media simples (comportamento original).

    Returns:
        team_strengths: DataFrame [team, attack, defense, matches, avg_for, avg_against]
        league_avgs:    dict {avg_home_goals, avg_away_goals, avg_total, metric, n_matches}
    """
    completed = matches_df[matches_df['status'] == 'complete'].copy()

    if use_xg:
        valid = completed.dropna(subset=['team_a_xg', 'team_b_xg'])
        col_home = 'team_a_xg'
        col_away = 'team_b_xg'
        metric = 'xG'
    else:
        valid = completed.dropna(subset=['homeGoalCount', 'awayGoalCount'])
        col_home = 'homeGoalCount'
        col_away = 'awayGoalCount'
        metric = 'Gols'

    # Medias da liga (sempre media simples — baseline para lambdas)
    avg_home = valid[col_home].mean()
    avg_away = valid[col_away].mean()

    league_avgs = {
        'avg_home_goals': avg_home,
        'avg_away_goals': avg_away,
        'avg_total': avg_home + avg_away,
        'metric': metric,
        'n_matches': len(valid),
    }

    if ema_alpha > 0:
        # EMA: jogos recentes pesam mais
        combined = _compute_ema_strengths(valid, col_home, col_away, ema_alpha)

        league_avg_for = combined['avg_for'].mean()
        league_avg_against = combined['avg_against'].mean()
    else:
        # Media simples (comportamento original)
        home_stats = valid.groupby('home_name').agg(
            home_matches=('home_name', 'count'),
            total_for_home=(col_home, 'sum'),
            total_against_home=(col_away, 'sum'),
        ).rename_axis('team')

        away_stats = valid.groupby('away_name').agg(
            away_matches=('away_name', 'count'),
            total_for_away=(col_away, 'sum'),
            total_against_away=(col_home, 'sum'),
        ).rename_axis('team')

        combined = home_stats.join(away_stats, how='outer').fillna(0)
        combined['matches'] = combined['home_matches'] + combined['away_matches']
        combined['total_for'] = combined['total_for_home'] + combined['total_for_away']
        combined['total_against'] = (combined['total_against_home']
                                     + combined['total_against_away'])
        combined['avg_for'] = combined['total_for'] / combined['matches']
        combined['avg_against'] = combined['total_against'] / combined['matches']

        league_avg_for = combined['total_for'].sum() / combined['matches'].sum()
        league_avg_against = (combined['total_against'].sum()
                              / combined['matches'].sum())

    # Forca relativa
    combined['attack'] = combined['avg_for'] / league_avg_for
    combined['defense'] = combined['avg_against'] / league_avg_against

    team_strengths = combined[
        ['attack', 'defense', 'matches', 'avg_for', 'avg_against']
    ].reset_index()

    return team_strengths, league_avgs


# ---------------------------------------------------------------------------
# 3. Calculo de lambdas (gols esperados)
# ---------------------------------------------------------------------------

def calculate_lambdas(home_team, away_team, team_strengths, league_avgs,
                      elo_ratings=None, hfa=HFA, elo_weight=ELO_LAMBDA_WEIGHT):
    """Calcula gols esperados (lambdas) para uma partida.

    Base Poisson:
        lambda_home = attack_home * defense_away * avg_goals_home_league
        lambda_away = attack_away * defense_home * avg_goals_away_league

    Ajuste ELO (se elo_ratings fornecido):
        elo_ratio = E_home / (1 - E_home)     # odds ratio do ELO
        lambda_home *= elo_ratio ^ elo_weight  # favorece mandante forte
        lambda_away *= (1/elo_ratio) ^ elo_weight

    Returns:
        (lambda_home, lambda_away)
    """
    strengths = team_strengths.set_index('team')

    # Lookup com fallback para time medio (1.0, 1.0)
    if home_team in strengths.index:
        atk_home = strengths.loc[home_team, 'attack']
        def_home = strengths.loc[home_team, 'defense']
    else:
        atk_home, def_home = 1.0, 1.0

    if away_team in strengths.index:
        atk_away = strengths.loc[away_team, 'attack']
        def_away = strengths.loc[away_team, 'defense']
    else:
        atk_away, def_away = 1.0, 1.0

    avg_h = league_avgs['avg_home_goals']
    avg_a = league_avgs['avg_away_goals']

    base_lambda_home = atk_home * def_away * avg_h
    base_lambda_away = atk_away * def_home * avg_a

    # Ajuste ELO
    if (elo_ratings is not None
            and home_team in elo_ratings
            and away_team in elo_ratings
            and elo_weight > 0):
        elo_h = elo_ratings[home_team]
        elo_a = elo_ratings[away_team]
        E_home = expected_score(elo_h, elo_a, hfa=hfa)
        E_home = max(0.01, min(0.99, E_home))

        elo_ratio = E_home / (1.0 - E_home)
        elo_adj_home = elo_ratio ** elo_weight
        elo_adj_away = (1.0 / elo_ratio) ** elo_weight

        lambda_home = base_lambda_home * elo_adj_home
        lambda_away = base_lambda_away * elo_adj_away
    else:
        lambda_home = base_lambda_home
        lambda_away = base_lambda_away

    # Clamp minimo
    lambda_home = max(0.2, lambda_home)
    lambda_away = max(0.2, lambda_away)

    return lambda_home, lambda_away


# ---------------------------------------------------------------------------
# 4. Simulacao de placares (Dixon-Coles)
# ---------------------------------------------------------------------------

def _dixon_coles_tau(h_goals, a_goals, lambda_h, lambda_a, rho):
    """Fator de correcao Dixon-Coles para placares baixos.

    Quando rho < 0 (tipico no futebol):
      - 0x0 e 1x1 ficam MAIS provaveis (empates)
      - 1x0 e 0x1 ficam MENOS provaveis
    Para todos os outros placares, tau = 1 (sem correcao).
    """
    if h_goals == 0 and a_goals == 0:
        return 1 - lambda_h * lambda_a * rho
    elif h_goals == 1 and a_goals == 0:
        return 1 + lambda_a * rho
    elif h_goals == 0 and a_goals == 1:
        return 1 + lambda_h * rho
    elif h_goals == 1 and a_goals == 1:
        return 1 - rho
    return 1.0


def score_probabilities(lambda_home, lambda_away, rho=DIXON_COLES_RHO,
                        max_goals=POISSON_MAX_GOALS):
    """Matriz de probabilidades com correcao Dixon-Coles.

    A correcao ajusta os 4 placares baixos (0x0, 1x0, 0x1, 1x1) via
    parametro rho, depois re-normaliza a matriz inteira.

    Returns:
        dict com matrix, home_win, draw, away_win,
        expected_home_goals, expected_away_goals
    """
    h_probs = poisson.pmf(range(max_goals + 1), lambda_home)
    a_probs = poisson.pmf(range(max_goals + 1), lambda_away)

    matrix = np.outer(h_probs, a_probs)

    # Aplicar correcao Dixon-Coles nos 4 placares baixos
    if rho != 0:
        for i in range(min(2, max_goals + 1)):
            for j in range(min(2, max_goals + 1)):
                tau = _dixon_coles_tau(i, j, lambda_home, lambda_away, rho)
                matrix[i, j] *= max(tau, 0.0)  # clamp para evitar prob negativa

        # Re-normalizar
        matrix /= matrix.sum()

    home_win = np.sum(np.tril(matrix, k=-1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, k=1))

    total = home_win + draw + away_win

    return {
        'matrix': matrix,
        'home_win': home_win / total,
        'draw': draw / total,
        'away_win': away_win / total,
        'expected_home_goals': lambda_home,
        'expected_away_goals': lambda_away,
    }


def simulate_score(lambda_home, lambda_away, rho=DIXON_COLES_RHO,
                   max_goals=POISSON_MAX_GOALS):
    """Simula um placar usando Dixon-Coles (ou Poisson puro se rho=0).

    Quando rho != 0, amostra da matriz corrigida ao inves de duas
    Poisson independentes.
    """
    if rho == 0:
        return (np.random.poisson(lambda_home), np.random.poisson(lambda_away))

    probs = score_probabilities(lambda_home, lambda_away, rho=rho,
                                max_goals=max_goals)
    flat = probs['matrix'].flatten()
    idx = np.random.choice(len(flat), p=flat)
    h = idx // (max_goals + 1)
    a = idx % (max_goals + 1)
    return (int(h), int(a))


# ---------------------------------------------------------------------------
# 4b. Blend modelo + odds
# ---------------------------------------------------------------------------

def blend_with_odds(model_p_home, model_p_draw, model_p_away,
                    odds_p_home, odds_p_draw, odds_p_away,
                    alpha=BLEND_ALPHA):
    """Combina probabilidades do modelo com probabilidades implicitas das odds.

    blend = alpha * modelo + (1 - alpha) * odds

    Calibrado na Fase 6: alpha=0.34 minimiza Brier Score na Serie A 2021-2026.
    Quando odds nao estao disponiveis, usar alpha=1.0 (modelo puro).

    Returns:
        (p_home, p_draw, p_away) blendados e normalizados
    """
    p_h = alpha * model_p_home + (1 - alpha) * odds_p_home
    p_d = alpha * model_p_draw + (1 - alpha) * odds_p_draw
    p_a = alpha * model_p_away + (1 - alpha) * odds_p_away

    # Re-normalizar (deve ser ~1.0, mas garantir)
    total = p_h + p_d + p_a
    return p_h / total, p_d / total, p_a / total


def odds_to_probs(odds_home, odds_draw, odds_away):
    """Converte odds decimais para probabilidades normalizadas (remove overround).

    P_implied(x) = 1 / odds(x)
    P_normalized(x) = P_implied(x) / sum(P_implied)
    """
    p_h = 1.0 / odds_home
    p_d = 1.0 / odds_draw
    p_a = 1.0 / odds_away
    total = p_h + p_d + p_a
    return p_h / total, p_d / total, p_a / total


# ---------------------------------------------------------------------------
# 5. Calibracao de rho (Dixon-Coles)
# ---------------------------------------------------------------------------

def calibrate_rho(matches_df, team_strengths, league_avgs, elo_ratings=None,
                  rho_range=np.arange(-0.25, 0.05, 0.01)):
    """Grid search para encontrar o rho que minimiza o Brier Score.

    Para cada rho, calcula probabilidades Dixon-Coles para cada jogo
    e computa o Brier Score multiclasse.

    Returns:
        (rho_otimo, results_df com colunas [rho, brier_score])
    """
    completed = matches_df[matches_df['status'] == 'complete'].copy()
    results = []

    # Pre-calcular lambdas (nao mudam com rho)
    lambdas = []
    actuals = []
    for _, match in completed.iterrows():
        lam_h, lam_a = calculate_lambdas(
            match['home_name'], match['away_name'],
            team_strengths, league_avgs, elo_ratings=elo_ratings,
        )
        lambdas.append((lam_h, lam_a))

        hg, ag = match['homeGoalCount'], match['awayGoalCount']
        if hg > ag:
            actuals.append((1, 0, 0))
        elif hg == ag:
            actuals.append((0, 1, 0))
        else:
            actuals.append((0, 0, 1))

    n = len(lambdas)

    for rho in rho_range:
        brier_sum = 0.0
        for (lam_h, lam_a), actual in zip(lambdas, actuals):
            probs = score_probabilities(lam_h, lam_a, rho=rho)
            p_h, p_d, p_a = probs['home_win'], probs['draw'], probs['away_win']
            brier_sum += ((p_h - actual[0])**2
                          + (p_d - actual[1])**2
                          + (p_a - actual[2])**2)
        results.append({'rho': round(rho, 3), 'brier_score': brier_sum / n})

    results_df = pd.DataFrame(results)
    best_idx = results_df['brier_score'].idxmin()
    rho_otimo = results_df.loc[best_idx, 'rho']

    return rho_otimo, results_df


# ---------------------------------------------------------------------------
# 6. Validacao
# ---------------------------------------------------------------------------

def validate_poisson(matches_df, team_strengths, league_avgs,
                     elo_ratings=None, label="Poisson"):
    """Valida o modelo contra jogos ja realizados.

    Calcula Brier Score e Log-Loss. Compara distribuicao prevista vs real.

    Returns:
        dict com metricas + predictions_df detalhado
    """
    completed = matches_df[matches_df['status'] == 'complete'].copy()

    brier_sum = 0.0
    log_loss_sum = 0.0
    n = 0
    sim_home, sim_away = [], []
    real_home, real_away = [], []
    predictions = []

    for _, match in completed.iterrows():
        home = match['home_name']
        away = match['away_name']

        lam_h, lam_a = calculate_lambdas(
            home, away, team_strengths, league_avgs, elo_ratings=elo_ratings,
        )

        probs = score_probabilities(lam_h, lam_a)
        p_h, p_d, p_a = probs['home_win'], probs['draw'], probs['away_win']

        hg = match['homeGoalCount']
        ag = match['awayGoalCount']
        if hg > ag:
            actual = (1, 0, 0)
        elif hg == ag:
            actual = (0, 1, 0)
        else:
            actual = (0, 0, 1)

        brier_sum += (p_h - actual[0])**2 + (p_d - actual[1])**2 + (p_a - actual[2])**2

        p_actual = p_h * actual[0] + p_d * actual[1] + p_a * actual[2]
        log_loss_sum += -np.log(max(p_actual, 1e-10))

        n += 1

        sg_h, sg_a = simulate_score(lam_h, lam_a)
        sim_home.append(sg_h)
        sim_away.append(sg_a)
        real_home.append(hg)
        real_away.append(ag)

        predictions.append({
            'home': home, 'away': away,
            'p_home': p_h, 'p_draw': p_d, 'p_away': p_a,
            'lambda_home': lam_h, 'lambda_away': lam_a,
            'goals_home': hg, 'goals_away': ag,
            'result': 'H' if hg > ag else ('D' if hg == ag else 'A'),
        })

    brier = brier_sum / n
    log_loss = log_loss_sum / n

    preds_df = pd.DataFrame(predictions)
    real_h_pct = (preds_df['result'] == 'H').mean()
    real_d_pct = (preds_df['result'] == 'D').mean()
    real_a_pct = (preds_df['result'] == 'A').mean()

    return {
        'label': label,
        'n_matches': n,
        'brier_score': brier,
        'log_loss': log_loss,
        'real_avg_goals': np.mean([h + a for h, a in zip(real_home, real_away)]),
        'sim_avg_goals': np.mean([h + a for h, a in zip(sim_home, sim_away)]),
        'real_avg_home': np.mean(real_home),
        'real_avg_away': np.mean(real_away),
        'sim_avg_home': np.mean(sim_home),
        'sim_avg_away': np.mean(sim_away),
        'real_home_win_pct': real_h_pct,
        'real_draw_pct': real_d_pct,
        'real_away_win_pct': real_a_pct,
        'pred_home_win_pct': preds_df['p_home'].mean(),
        'pred_draw_pct': preds_df['p_draw'].mean(),
        'pred_away_win_pct': preds_df['p_away'].mean(),
        'predictions_df': preds_df,
    }


def validate_poisson_dc(matches_df, team_strengths, league_avgs,
                        elo_ratings=None, rho=DIXON_COLES_RHO,
                        label="Dixon-Coles"):
    """Igual a validate_poisson mas com rho explicito (para testar varios rhos)."""
    completed = matches_df[matches_df['status'] == 'complete'].copy()

    brier_sum = 0.0
    log_loss_sum = 0.0
    n = 0
    sim_home, sim_away = [], []
    real_home, real_away = [], []
    predictions = []

    for _, match in completed.iterrows():
        home = match['home_name']
        away = match['away_name']

        lam_h, lam_a = calculate_lambdas(
            home, away, team_strengths, league_avgs, elo_ratings=elo_ratings,
        )

        probs = score_probabilities(lam_h, lam_a, rho=rho)
        p_h, p_d, p_a = probs['home_win'], probs['draw'], probs['away_win']

        hg = match['homeGoalCount']
        ag = match['awayGoalCount']
        if hg > ag:
            actual = (1, 0, 0)
        elif hg == ag:
            actual = (0, 1, 0)
        else:
            actual = (0, 0, 1)

        brier_sum += (p_h - actual[0])**2 + (p_d - actual[1])**2 + (p_a - actual[2])**2

        p_actual = p_h * actual[0] + p_d * actual[1] + p_a * actual[2]
        log_loss_sum += -np.log(max(p_actual, 1e-10))

        n += 1

        sg_h, sg_a = simulate_score(lam_h, lam_a, rho=rho)
        sim_home.append(sg_h)
        sim_away.append(sg_a)
        real_home.append(hg)
        real_away.append(ag)

        predictions.append({
            'home': home, 'away': away,
            'p_home': p_h, 'p_draw': p_d, 'p_away': p_a,
            'lambda_home': lam_h, 'lambda_away': lam_a,
            'goals_home': hg, 'goals_away': ag,
            'result': 'H' if hg > ag else ('D' if hg == ag else 'A'),
        })

    brier = brier_sum / n
    log_loss = log_loss_sum / n

    preds_df = pd.DataFrame(predictions)
    real_h_pct = (preds_df['result'] == 'H').mean()
    real_d_pct = (preds_df['result'] == 'D').mean()
    real_a_pct = (preds_df['result'] == 'A').mean()

    return {
        'label': label,
        'n_matches': n,
        'brier_score': brier,
        'log_loss': log_loss,
        'real_avg_goals': np.mean([h + a for h, a in zip(real_home, real_away)]),
        'sim_avg_goals': np.mean([h + a for h, a in zip(sim_home, sim_away)]),
        'real_avg_home': np.mean(real_home),
        'real_avg_away': np.mean(real_away),
        'sim_avg_home': np.mean(sim_home),
        'sim_avg_away': np.mean(sim_away),
        'real_home_win_pct': real_h_pct,
        'real_draw_pct': real_d_pct,
        'real_away_win_pct': real_a_pct,
        'pred_home_win_pct': preds_df['p_home'].mean(),
        'pred_draw_pct': preds_df['p_draw'].mean(),
        'pred_away_win_pct': preds_df['p_away'].mean(),
        'predictions_df': preds_df,
    }


def validate_elo_only(matches_df, elo_ratings, hfa=HFA):
    """Brier Score do ELO puro (sem Poisson) como baseline.

    Usa modelo tripartido da Fase 2:
      P_draw = DRAW_BASE_RATE * (1 - |elo_diff| / 800)
      P_home = E_home * (1 - P_draw)
      P_away = (1 - E_home) * (1 - P_draw)
    """
    completed = matches_df[matches_df['status'] == 'complete'].copy()

    brier_sum = 0.0
    n = 0

    for _, match in completed.iterrows():
        home = match['home_name']
        away = match['away_name']

        elo_h = elo_ratings.get(home, 1500)
        elo_a = elo_ratings.get(away, 1500)

        E_home = expected_score(elo_h, elo_a, hfa=hfa)

        elo_diff = abs((elo_h + hfa) - elo_a)
        p_draw = DRAW_BASE_RATE * (1 - elo_diff / 800)
        p_draw = max(0.05, min(0.40, p_draw))

        p_home = E_home * (1 - p_draw)
        p_away = (1 - E_home) * (1 - p_draw)

        hg = match['homeGoalCount']
        ag = match['awayGoalCount']
        if hg > ag:
            actual = (1, 0, 0)
        elif hg == ag:
            actual = (0, 1, 0)
        else:
            actual = (0, 0, 1)

        brier_sum += (p_home - actual[0])**2 + (p_draw - actual[1])**2 + (p_away - actual[2])**2
        n += 1

    return brier_sum / n if n > 0 else None


# ---------------------------------------------------------------------------
# __main__: roda pipeline completo da Fase 3
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.elo import load_historical_matches, calculate_all_elos

    print("=" * 60)
    print("BRForecast — Fase 3: Modelo Poisson + xG")
    print("=" * 60)

    # --- 1. Carregar dados ---
    print("\n1. Carregando partidas Serie A com xG...")
    matches_xg = load_serie_a_with_xg()
    completed_all = matches_xg[matches_xg['status'] == 'complete']
    n_total = len(completed_all)
    n_xg = completed_all.dropna(subset=['team_a_xg', 'team_b_xg']).shape[0]
    print(f"   {n_total} partidas completas, {n_xg} com xG ({100*n_xg/n_total:.1f}%)")
    print(f"   Temporadas: {sorted(completed_all['season_year'].unique())}")

    # --- 2. Forcas ofensivas/defensivas (temporada alvo) ---
    print(f"\n2. Calculando forcas ofensivas/defensivas (temporada {TARGET_YEAR})...")
    target = matches_xg[matches_xg['season_year'] == TARGET_YEAR]
    target_done = target[target['status'] == 'complete']

    if len(target_done) < MIN_MATCHES_SEASON:
        print(f"   AVISO: apenas {len(target_done)} jogos em {TARGET_YEAR} "
              f"(minimo={MIN_MATCHES_SEASON}). Incluindo {TARGET_YEAR - 1}.")
        expanded = matches_xg[matches_xg['season_year'] >= TARGET_YEAR - 1]
        team_strengths, league_avgs = calculate_team_strengths(expanded)
    else:
        team_strengths, league_avgs = calculate_team_strengths(target_done)

    print(f"   Metrica: {league_avgs['metric']}")
    print(f"   Partidas usadas: {league_avgs['n_matches']}")
    print(f"   Media liga: mandante={league_avgs['avg_home_goals']:.3f}, "
          f"visitante={league_avgs['avg_away_goals']:.3f}")
    print(f"   Times: {len(team_strengths)}")

    top_atk = team_strengths.nlargest(5, 'attack')
    top_def_good = team_strengths.nsmallest(5, 'defense')
    top_def_bad = team_strengths.nlargest(5, 'defense')

    print(f"\n   === Top 5 Ataque (attack > 1.0 = acima da media) ===")
    for _, r in top_atk.iterrows():
        print(f"   {r['team']:25s} attack={r['attack']:.3f}  avg_xG={r['avg_for']:.2f}")

    print(f"\n   === Top 5 Defesa - melhores (defense < 1.0 = sofre menos) ===")
    for _, r in top_def_good.iterrows():
        print(f"   {r['team']:25s} defense={r['defense']:.3f}  avg_xGA={r['avg_against']:.2f}")

    print(f"\n   === Top 5 Defesa - piores (defense > 1.0 = sofre mais) ===")
    for _, r in top_def_bad.iterrows():
        print(f"   {r['team']:25s} defense={r['defense']:.3f}  avg_xGA={r['avg_against']:.2f}")

    # --- 3. Carregar ELOs ---
    print(f"\n3. Carregando ELOs...")
    elo_matches = load_historical_matches()
    elo_ratings, _ = calculate_all_elos(elo_matches)
    print(f"   {len(elo_ratings)} times com ELO")

    # --- 4. Calibrar rho (Dixon-Coles) ---
    print(f"\n4. Calibrando rho (Dixon-Coles) via grid search...")

    # Usar ultima temporada completa
    for vy in [TARGET_YEAR - 1, TARGET_YEAR - 2]:
        val_matches = matches_xg[matches_xg['season_year'] == vy]
        val_done = val_matches[val_matches['status'] == 'complete']
        if len(val_done) >= 300:
            break

    print(f"   Temporada de calibracao: {vy} ({len(val_done)} jogos)")
    val_strengths, val_avgs = calculate_team_strengths(val_done)

    rho_otimo, rho_results = calibrate_rho(
        val_done, val_strengths, val_avgs,
        elo_ratings=elo_ratings,
        rho_range=np.arange(-0.25, 0.05, 0.01),
    )
    print(f"   Rho otimo: {rho_otimo:.2f}")
    print(f"   Brier no otimo: {rho_results.loc[rho_results['brier_score'].idxmin(), 'brier_score']:.5f}")
    print(f"   Brier sem DC (rho=0): "
          f"{rho_results.loc[rho_results['rho'] == 0.0, 'brier_score'].values[0]:.5f}")

    # --- 5. Teste sintetico: lambda conhecido ---
    print(f"\n5. Teste sintetico (lambda_home=1.5, lambda_away=1.0, n=10.000)...")
    np.random.seed(42)
    N_TEST = 10000
    test_results = [simulate_score(1.5, 1.0, rho=rho_otimo) for _ in range(N_TEST)]
    hw = sum(1 for h, a in test_results if h > a)
    dr = sum(1 for h, a in test_results if h == a)
    aw = sum(1 for h, a in test_results if h < a)
    avg_g = np.mean([h + a for h, a in test_results])

    print(f"   Mandante vence: {100*hw/N_TEST:.1f}%")
    print(f"   Empates:        {100*dr/N_TEST:.1f}%")
    print(f"   Visitante vence:{100*aw/N_TEST:.1f}%")
    print(f"   Media de gols:  {avg_g:.2f}")

    probs_pure = score_probabilities(1.5, 1.0, rho=0)
    probs_dc = score_probabilities(1.5, 1.0, rho=rho_otimo)
    print(f"\n   Poisson puro:    P(H)={probs_pure['home_win']:.3f}  "
          f"P(D)={probs_pure['draw']:.3f}  P(A)={probs_pure['away_win']:.3f}")
    print(f"   Dixon-Coles:     P(H)={probs_dc['home_win']:.3f}  "
          f"P(D)={probs_dc['draw']:.3f}  P(A)={probs_dc['away_win']:.3f}")

    # --- 6. Validacao em temporada completa ---
    print(f"\n6. Validacao em temporada completa ({vy}, {len(val_done)} jogos)...")

    # a) ELO puro (baseline)
    brier_elo = validate_elo_only(val_done, elo_ratings)

    # b) Poisson+ELO sem Dixon-Coles (rho=0)
    val_no_dc = validate_poisson_dc(val_done, val_strengths, val_avgs,
                                    elo_ratings=elo_ratings, rho=0,
                                    label="Poisson+ELO (sem DC)")

    # c) Poisson+ELO com Dixon-Coles (rho calibrado)
    val_dc = validate_poisson_dc(val_done, val_strengths, val_avgs,
                                 elo_ratings=elo_ratings, rho=rho_otimo,
                                 label="Dixon-Coles+ELO")

    print(f"\n   {'='*55}")
    print(f"   {'Modelo':<25s} {'Brier Score':>12s} {'Log-Loss':>10s}")
    print(f"   {'-'*55}")
    print(f"   {'ELO puro':<25s} {brier_elo:>12.5f} {'N/A':>10s}")
    print(f"   {'Poisson+ELO (sem DC)':<25s} {val_no_dc['brier_score']:>12.5f} {val_no_dc['log_loss']:>10.4f}")
    print(f"   {'Dixon-Coles+ELO':<25s} {val_dc['brier_score']:>12.5f} {val_dc['log_loss']:>10.4f}")
    print(f"   {'='*55}")

    best = min(
        [("ELO puro", brier_elo),
         ("Poisson+ELO (sem DC)", val_no_dc['brier_score']),
         ("Dixon-Coles+ELO", val_dc['brier_score'])],
        key=lambda x: x[1],
    )
    print(f"   Melhor: {best[0]} (BS={best[1]:.5f})")

    # Distribuicao de resultados
    print(f"\n   === Distribuicao de resultados ({vy}) ===")
    print(f"   {'':22s} {'Real':>8s} {'sem DC':>10s} {'Dixon-Coles':>12s}")
    print(f"   {'Mandante vence':22s} {val_dc['real_home_win_pct']:>7.1%} "
          f"{val_no_dc['pred_home_win_pct']:>9.1%} {val_dc['pred_home_win_pct']:>11.1%}")
    print(f"   {'Empate':22s} {val_dc['real_draw_pct']:>7.1%} "
          f"{val_no_dc['pred_draw_pct']:>9.1%} {val_dc['pred_draw_pct']:>11.1%}")
    print(f"   {'Visitante vence':22s} {val_dc['real_away_win_pct']:>7.1%} "
          f"{val_no_dc['pred_away_win_pct']:>9.1%} {val_dc['pred_away_win_pct']:>11.1%}")
    print(f"   {'Gols medios':22s} {val_dc['real_avg_goals']:>7.2f} "
          f"{val_no_dc['sim_avg_goals']:>9.2f} {val_dc['sim_avg_goals']:>11.2f}")

    # --- 6. Exportar ---
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"team_strengths_{TARGET_YEAR}.csv")
    team_strengths.to_csv(path, index=False)
    print(f"\n6. Exportado: {path}")

    print(f"\nFase 3 concluida.")
