"""BRForecast — Match Importance (importancia de cada jogo).

Mede quanto o resultado de um jogo impacta as probabilidades de cada time.
Inspirado no modelo FiveThirtyEight:
  - Jogos mais importantes para um time fazem ele render acima do esperado
  - Jogos sem importancia para ambos tem mais variancia

Calculo:
  Para cada jogo futuro, roda mini-simulacoes com resultado fixado em
  vitoria/empate/derrota. A importance e o delta maximo nas probabilidades
  de titulo, Libertadores ou rebaixamento entre os cenarios.

  importance_team = max(|P(zona|vitoria) - P(zona|derrota)|)
                    para zona em {titulo, libertadores, rebaixamento}

Uso:
  - Multiplicar lambdas por (1 + IMPORTANCE_LAMBDA_BOOST * importance_norm)
    para o time com mais a perder
  - Aumentar variancia quando importance e baixa para ambos
"""

import numpy as np
import pandas as pd

from src.config import (
    IMPORTANCE_N_SIMS, IMPORTANCE_LAMBDA_BOOST,
    HFA, ELO_LAMBDA_WEIGHT, POISSON_MAX_GOALS, DIXON_COLES_RHO,
    K_SIMULATION, ZONES,
)
from src.table import table_to_arrays, update_table_np, rank_teams_np, PTS
from src.simulation import _prepare_matches_hot, _simulate_season_hot


# ---------------------------------------------------------------------------
# 1. Mini Monte Carlo para um cenario fixo
# ---------------------------------------------------------------------------

def _mini_monte_carlo(n_sims, remaining_matches, current_table,
                      team_strengths, league_avgs, elo_ratings,
                      fixed_match_idx, fixed_result):
    """Roda mini-simulacao com UM jogo fixado.

    Args:
        fixed_match_idx: indice (no remaining_matches) do jogo a fixar
        fixed_result: (goals_home, goals_away) fixados

    Returns:
        dict {team: {p_titulo, p_libertadores, p_rebaixamento}}
    """
    base_data, team_names, team_idx = table_to_arrays(current_table)
    n_teams = len(team_names)

    # Separar jogo fixado dos demais
    remaining_list = list(remaining_matches.iterrows())
    fixed_row = remaining_list[fixed_match_idx][1]
    fixed_h_idx = team_idx.get(fixed_row['mandante'], -1)
    fixed_a_idx = team_idx.get(fixed_row['visitante'], -1)

    free_rows = [r for i, (_, r) in enumerate(remaining_list) if i != fixed_match_idx]
    free_df = pd.DataFrame(free_rows) if free_rows else pd.DataFrame()

    if len(free_df) == 0 or elo_ratings is None:
        return {}

    matches_info, strengths_indexed = _prepare_matches_hot(
        free_df, team_strengths, team_idx,
    )

    # Contadores de posicoes
    pos_counts = np.zeros((n_teams, 3), dtype=np.int32)  # titulo, liberta, rebx

    for _ in range(n_sims):
        data = base_data.copy()
        elo_sim = dict(elo_ratings)

        # Aplicar resultado fixo
        update_table_np(data, fixed_h_idx, fixed_a_idx,
                        fixed_result[0], fixed_result[1])

        # Hot ELO para o jogo fixado
        elo_h = elo_sim.get(fixed_row['mandante'], 1500)
        elo_a = elo_sim.get(fixed_row['visitante'], 1500)
        exp_h = 1.0 / (1.0 + 10.0 ** ((elo_a - (elo_h + HFA)) / 400.0))
        s_h = 1.0 if fixed_result[0] > fixed_result[1] else (
            0.5 if fixed_result[0] == fixed_result[1] else 0.0)
        elo_sim[fixed_row['mandante']] = elo_h + K_SIMULATION * (s_h - exp_h)
        elo_sim[fixed_row['visitante']] = elo_a + K_SIMULATION * ((1.0 - s_h) - (1.0 - exp_h))

        # Simular restante
        positions = _simulate_season_hot(
            data, matches_info, strengths_indexed,
            league_avgs, elo_sim,
            hfa=HFA, k_sim=K_SIMULATION,
            elo_weight=ELO_LAMBDA_WEIGHT,
            rho=DIXON_COLES_RHO, max_goals=POISSON_MAX_GOALS,
        )

        for i in range(n_teams):
            if positions[i] == 1:
                pos_counts[i, 0] += 1
            if positions[i] <= 6:
                pos_counts[i, 1] += 1
            if positions[i] >= 17:
                pos_counts[i, 2] += 1

    result = {}
    for i, team in enumerate(team_names):
        result[team] = {
            'p_titulo': pos_counts[i, 0] / n_sims,
            'p_libertadores': pos_counts[i, 1] / n_sims,
            'p_rebaixamento': pos_counts[i, 2] / n_sims,
        }
    return result


# ---------------------------------------------------------------------------
# 2. Calculo de importance por jogo
# ---------------------------------------------------------------------------

def calculate_match_importance(match_idx, remaining_matches, current_table,
                               team_strengths, league_avgs, elo_ratings,
                               n_sims=IMPORTANCE_N_SIMS):
    """Calcula a importance de um jogo para ambos os times.

    Roda 3 cenarios: vitoria mandante (2-0), empate (1-1), vitoria visitante (0-2).
    Importance = max delta entre cenarios nas probabilidades-chave.

    Returns:
        dict {
            'home': nome mandante,
            'away': nome visitante,
            'importance_home': float 0-1,
            'importance_away': float 0-1,
            'importance_match': float 0-1 (media harmonica),
            'details': dict com probabilidades por cenario
        }
    """
    remaining_list = list(remaining_matches.iterrows())
    match_row = remaining_list[match_idx][1]
    home = match_row['mandante']
    away = match_row['visitante']

    scenarios = {
        'home_win': (2, 0),
        'draw': (1, 1),
        'away_win': (0, 2),
    }

    scenario_results = {}
    for scenario_name, result in scenarios.items():
        scenario_results[scenario_name] = _mini_monte_carlo(
            n_sims, remaining_matches, current_table,
            team_strengths, league_avgs, elo_ratings,
            match_idx, result,
        )

    if not scenario_results.get('home_win') or not scenario_results.get('away_win'):
        return {
            'home': home, 'away': away,
            'importance_home': 0.0, 'importance_away': 0.0,
            'importance_match': 0.0, 'details': {},
        }

    # Importance = max delta entre vitoria e derrota
    hw = scenario_results['home_win']
    aw = scenario_results['away_win']

    def team_importance(team):
        deltas = []
        for key in ['p_titulo', 'p_libertadores', 'p_rebaixamento']:
            p_best = hw.get(team, {}).get(key, 0)
            p_worst = aw.get(team, {}).get(key, 0)
            deltas.append(abs(p_best - p_worst))
        return max(deltas) if deltas else 0.0

    imp_home = team_importance(home)
    imp_away = team_importance(away)

    # Media harmonica (como o 538)
    if imp_home + imp_away > 0:
        imp_match = 2 * imp_home * imp_away / (imp_home + imp_away)
    else:
        imp_match = 0.0

    return {
        'home': home,
        'away': away,
        'importance_home': round(imp_home, 4),
        'importance_away': round(imp_away, 4),
        'importance_match': round(imp_match, 4),
        'details': scenario_results,
    }


# ---------------------------------------------------------------------------
# 3. Importance para todos os jogos restantes
# ---------------------------------------------------------------------------

def calculate_all_importance(remaining_matches, current_table,
                             team_strengths, league_avgs, elo_ratings,
                             n_sims=IMPORTANCE_N_SIMS, max_matches=None,
                             show_progress=True):
    """Calcula importance para cada jogo restante.

    Args:
        max_matches: se definido, calcula apenas para os proximos N jogos
                     (util para performance — proxima rodada apenas)

    Returns:
        DataFrame [home, away, rodada, importance_home, importance_away,
                   importance_match]
    """
    from tqdm import tqdm

    remaining_list = list(remaining_matches.iterrows())
    n_total = min(len(remaining_list), max_matches) if max_matches else len(remaining_list)

    records = []
    iterator = range(n_total)
    if show_progress:
        iterator = tqdm(iterator, desc="Calculando importance", unit="jogo")

    for i in iterator:
        match_row = remaining_list[i][1]
        result = calculate_match_importance(
            i, remaining_matches, current_table,
            team_strengths, league_avgs, elo_ratings,
            n_sims=n_sims,
        )
        result['rodada'] = match_row.get('rodada')
        records.append({
            'home': result['home'],
            'away': result['away'],
            'rodada': result['rodada'],
            'importance_home': result['importance_home'],
            'importance_away': result['importance_away'],
            'importance_match': result['importance_match'],
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Ajuste de lambdas por importance
# ---------------------------------------------------------------------------

def adjust_lambdas_importance(lambda_home, lambda_away,
                              importance_home, importance_away,
                              boost=IMPORTANCE_LAMBDA_BOOST):
    """Ajusta lambdas com base na importance do jogo.

    Time com jogo decisivo rende mais: lambda *= (1 + boost * importance)
    Quando ambos tem importance baixa, nao muda.

    Args:
        lambda_home, lambda_away: lambdas base (Poisson)
        importance_home, importance_away: 0 a 1
        boost: fator maximo de boost (default 0.05 = +5%)

    Returns:
        (lambda_home_adj, lambda_away_adj)
    """
    adj_h = lambda_home * (1.0 + boost * importance_home)
    adj_a = lambda_away * (1.0 + boost * importance_away)
    return adj_h, adj_a


# ---------------------------------------------------------------------------
# __main__: diagnostico
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.elo import load_historical_matches, calculate_all_elos
    from src.poisson import load_serie_a_with_xg, calculate_team_strengths
    from src.load_data import get_season_id, load_current_table, load_remaining_matches
    from src.config import TARGET_YEAR, MIN_MATCHES_SEASON

    print("=" * 60)
    print("BRForecast — Match Importance")
    print("=" * 60)

    # Carregar dados
    print("\n1. Carregando dados...")
    season_id = get_season_id(TARGET_YEAR)
    table = load_current_table(season_id)
    remaining = load_remaining_matches(season_id)
    print(f"   {len(table)} times, {len(remaining)} jogos restantes")

    # ELO
    elo_matches = load_historical_matches()
    elo_ratings, _ = calculate_all_elos(elo_matches)

    # Forcas Poisson
    matches_xg = load_serie_a_with_xg()
    target_m = matches_xg[matches_xg['season_year'] == TARGET_YEAR]
    target_done = target_m[target_m['status'] == 'complete']
    if len(target_done) < MIN_MATCHES_SEASON:
        expanded = matches_xg[matches_xg['season_year'] >= TARGET_YEAR - 1]
        team_strengths, league_avgs = calculate_team_strengths(expanded)
    else:
        team_strengths, league_avgs = calculate_team_strengths(target_done)

    # Calcular importance para proxima rodada
    next_round = remaining['rodada'].min()
    next_matches = remaining[remaining['rodada'] == next_round]
    print(f"\n2. Calculando importance para rodada {next_round} "
          f"({len(next_matches)} jogos, {IMPORTANCE_N_SIMS} sims cada)...")

    # Usar apenas jogos da proxima rodada
    imp_df = calculate_all_importance(
        next_matches, table, team_strengths, league_avgs, elo_ratings,
        n_sims=IMPORTANCE_N_SIMS,
    )

    print(f"\n   === Importance — Rodada {next_round} ===")
    print(f"   {'Jogo':45s} {'Imp H':>6s} {'Imp A':>6s} {'Match':>6s}")
    print(f"   {'-'*65}")
    imp_sorted = imp_df.sort_values('importance_match', ascending=False)
    for _, r in imp_sorted.iterrows():
        jogo = f"{r['home']} vs {r['away']}"
        print(f"   {jogo:45s} {r['importance_home']:>5.1%} "
              f"{r['importance_away']:>5.1%} {r['importance_match']:>5.1%}")

    # Exemplo de ajuste de lambda
    if len(imp_sorted) > 0:
        top = imp_sorted.iloc[0]
        print(f"\n3. Exemplo de ajuste de lambda (jogo mais importante):")
        print(f"   {top['home']} vs {top['away']}")
        lam_h, lam_a = 1.4, 1.1  # exemplo
        adj_h, adj_a = adjust_lambdas_importance(
            lam_h, lam_a, top['importance_home'], top['importance_away'],
        )
        print(f"   Lambda original:  home={lam_h:.3f}, away={lam_a:.3f}")
        print(f"   Lambda ajustado:  home={adj_h:.3f}, away={adj_a:.3f}")
        print(f"   Boost: home={adj_h/lam_h - 1:+.2%}, away={adj_a/lam_a - 1:+.2%}")
