"""BRForecast — Simulação Monte Carlo do campeonato.

Simula o restante do campeonato N vezes usando o modelo Poisson/Dixon-Coles.

Dois modos de simulação:
  1. Estático (hot_update=False): Pré-computa CDFs fixas. Rápido (~30s/20k sims).
  2. Hot Update (hot_update=True): Recalcula lambdas a cada jogo simulado usando
     ELO atualizado. Mais realista, captura momentum. (~3-5 min/20k sims).

Hot Update: após cada jogo simulado, atualiza o ELO de ambos os times
com K_SIMULATION fixo. Os lambdas dos jogos seguintes usam o ELO atualizado,
mudando as probabilidades conforme times ganham/perdem forma na simulação.
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.stats import poisson as poisson_dist

from src.config import (
    DB_PATH, DATA_DIR, SERIE_A_IDS, TARGET_YEAR,
    HFA, ELO_LAMBDA_WEIGHT, ZONES, N_SIMULATIONS,
    MIN_MATCHES_SEASON, POISSON_MAX_GOALS, DIXON_COLES_RHO,
    HOT_UPDATE, K_SIMULATION,
)
from src.table import (
    table_to_arrays, update_table_np, rank_teams_np,
    update_table, apply_tiebreakers,
)
from src.poisson import calculate_lambdas, score_probabilities
from src.elo import expected_score


# ---------------------------------------------------------------------------
# 1. Pré-computação de lambdas e CDFs
# ---------------------------------------------------------------------------

def _precompute_match_cdfs(remaining_matches, team_strengths, league_avgs,
                           elo_ratings, team_idx):
    """Pré-computa lambdas e CDFs cumulativas para todos os jogos restantes.

    Returns:
        match_h_idx: ndarray (n_matches,) — índice do mandante no array
        match_a_idx: ndarray (n_matches,) — índice do visitante no array
        cdfs: ndarray (n_matches, (max_goals+1)^2) — CDF acumulada por jogo
        score_h: ndarray ((max_goals+1)^2,) — gols mandante para cada idx
        score_a: ndarray ((max_goals+1)^2,) — gols visitante para cada idx
    """
    mg = POISSON_MAX_GOALS
    n_scores = (mg + 1) ** 2
    n_matches = len(remaining_matches)

    match_h_idx = np.zeros(n_matches, dtype=np.int32)
    match_a_idx = np.zeros(n_matches, dtype=np.int32)
    cdfs = np.zeros((n_matches, n_scores), dtype=np.float64)

    # Tabela de lookup: idx -> (gols_h, gols_a)
    score_h = np.zeros(n_scores, dtype=np.int32)
    score_a = np.zeros(n_scores, dtype=np.int32)
    for idx in range(n_scores):
        score_h[idx] = idx // (mg + 1)
        score_a[idx] = idx % (mg + 1)

    for i, (_, match) in enumerate(remaining_matches.iterrows()):
        home = match['mandante']
        away = match['visitante']

        match_h_idx[i] = team_idx.get(home, -1)
        match_a_idx[i] = team_idx.get(away, -1)

        lam_h, lam_a = calculate_lambdas(
            home, away, team_strengths, league_avgs,
            elo_ratings=elo_ratings,
        )

        probs = score_probabilities(lam_h, lam_a)
        flat = probs['matrix'].flatten()

        # CDF acumulada para sampling via searchsorted
        cdfs[i] = np.cumsum(flat)
        # Garantir que termine em 1.0 exatamente
        cdfs[i] /= cdfs[i, -1]

    return match_h_idx, match_a_idx, cdfs, score_h, score_a


# ---------------------------------------------------------------------------
# 2. Simulação de uma temporada (numpy)
# ---------------------------------------------------------------------------

def _simulate_season_np(table_data, match_h_idx, match_a_idx,
                        cdfs, score_h, score_a):
    """Simula uma temporada usando arrays numpy pré-computados.

    Modifica table_data in-place.
    Returns: positions array (1-based)
    """
    n_matches = len(match_h_idx)

    # Gerar números aleatórios para todos os jogos de uma vez
    rands = np.random.random(n_matches)

    for i in range(n_matches):
        # Amostrar placar via CDF
        idx = np.searchsorted(cdfs[i], rands[i])
        h_goals = int(score_h[idx])
        a_goals = int(score_a[idx])

        update_table_np(table_data, match_h_idx[i], match_a_idx[i],
                        h_goals, a_goals)

    return rank_teams_np(table_data)


# ---------------------------------------------------------------------------
# 2b. Preparação e simulação HOT (ELO + Lambda atualizados a cada jogo)
# ---------------------------------------------------------------------------

def _prepare_matches_hot(remaining_matches, team_strengths, team_idx):
    """Prepara dados dos jogos para simulação hot.

    Pré-indexa forças ofensivas/defensivas para evitar lookups por nome
    no loop interno.

    Returns:
        matches_info: list of tuples (h_idx, a_idx, home_name, away_name)
        strengths_indexed: dict {team_name: (attack, defense)}
    """
    strengths_indexed = {}
    ts = team_strengths.set_index('team')
    for team in ts.index:
        strengths_indexed[team] = (
            float(ts.loc[team, 'attack']),
            float(ts.loc[team, 'defense']),
        )

    matches_info = []
    for _, match in remaining_matches.iterrows():
        home = match['mandante']
        away = match['visitante']
        h_idx = team_idx.get(home, -1)
        a_idx = team_idx.get(away, -1)
        matches_info.append((h_idx, a_idx, home, away))

    return matches_info, strengths_indexed


def _simulate_season_hot(table_data, matches_info, strengths_indexed,
                         league_avgs, elo_ratings_sim, hfa, k_sim,
                         elo_weight, rho, max_goals):
    """Simula uma temporada com Hot ELO + Hot Lambda.

    Após cada jogo simulado:
      1. Atualiza ELO de ambos os times
      2. Os lambdas dos jogos seguintes usam ELO atualizado

    Args:
        table_data: numpy array (n_teams, 8) — modificado in-place
        matches_info: list of (h_idx, a_idx, home_name, away_name)
        strengths_indexed: dict {team: (attack, defense)}
        league_avgs: dict com avg_home_goals, avg_away_goals
        elo_ratings_sim: dict {team: elo} — CÓPIA, modificada in-place
        hfa: Home Field Advantage
        k_sim: K factor para atualização do ELO
        elo_weight: peso do ELO no ajuste dos lambdas
        rho: Dixon-Coles rho
        max_goals: máximo de gols na matriz
    """
    avg_h = league_avgs['avg_home_goals']
    avg_a = league_avgs['avg_away_goals']
    goal_range = np.arange(max_goals + 1)
    mg1 = max_goals + 1

    for h_idx, a_idx, home, away in matches_info:
        # Forças ofensivas/defensivas (fixas, baseadas em xG real)
        atk_h, def_h = strengths_indexed.get(home, (1.0, 1.0))
        atk_a, def_a = strengths_indexed.get(away, (1.0, 1.0))

        # Lambdas base
        base_lam_h = atk_h * def_a * avg_h
        base_lam_a = atk_a * def_h * avg_a

        # Ajuste ELO (usando ratings atualizados)
        elo_h = elo_ratings_sim.get(home, 1500)
        elo_a = elo_ratings_sim.get(away, 1500)

        if elo_weight > 0:
            E_home = 1.0 / (1.0 + 10.0 ** ((elo_a - (elo_h + hfa)) / 400.0))
            E_home = max(0.01, min(0.99, E_home))
            elo_ratio = E_home / (1.0 - E_home)
            lam_h = max(0.2, base_lam_h * (elo_ratio ** elo_weight))
            lam_a = max(0.2, base_lam_a * ((1.0 / elo_ratio) ** elo_weight))
        else:
            lam_h = max(0.2, base_lam_h)
            lam_a = max(0.2, base_lam_a)

        # Matriz de probabilidades (Poisson + Dixon-Coles inline)
        h_pmf = poisson_dist.pmf(goal_range, lam_h)
        a_pmf = poisson_dist.pmf(goal_range, lam_a)
        matrix = np.outer(h_pmf, a_pmf)

        if rho != 0:
            matrix[0, 0] *= max(0.0, 1.0 - lam_h * lam_a * rho)
            matrix[1, 0] *= max(0.0, 1.0 + lam_a * rho)
            matrix[0, 1] *= max(0.0, 1.0 + lam_h * rho)
            matrix[1, 1] *= max(0.0, 1.0 - rho)
            matrix /= matrix.sum()

        # Amostrar placar via CDF
        flat = matrix.flatten()
        cdf = np.cumsum(flat)
        cdf /= cdf[-1]
        idx = np.searchsorted(cdf, np.random.random())
        h_goals = int(idx // mg1)
        a_goals = int(idx % mg1)

        # Atualizar tabela
        update_table_np(table_data, h_idx, a_idx, h_goals, a_goals)

        # Hot ELO update
        exp_home = 1.0 / (1.0 + 10.0 ** ((elo_a - (elo_h + hfa)) / 400.0))
        if h_goals > a_goals:
            score_h, score_a = 1.0, 0.0
        elif h_goals == a_goals:
            score_h, score_a = 0.5, 0.5
        else:
            score_h, score_a = 0.0, 1.0

        elo_ratings_sim[home] = elo_h + k_sim * (score_h - exp_home)
        elo_ratings_sim[away] = elo_a + k_sim * (score_a - (1.0 - exp_home))

    return rank_teams_np(table_data)


# ---------------------------------------------------------------------------
# 3. Monte Carlo
# ---------------------------------------------------------------------------

def run_monte_carlo(n_simulations, remaining_matches, current_table,
                    team_strengths, league_avgs, elo_ratings=None,
                    show_progress=True, hot_update=HOT_UPDATE):
    """Roda N simulações do campeonato.

    Args:
        hot_update: Se True, atualiza ELO e recalcula lambdas a cada jogo
                    simulado (mais realista, ~10x mais lento).

    Returns:
        positions: ndarray (n_sims, n_teams)
        points: ndarray (n_sims, n_teams)
        team_names: list
    """
    # Converter tabela para arrays
    base_data, team_names, team_idx = table_to_arrays(current_table)
    n_teams = len(team_names)

    positions = np.zeros((n_simulations, n_teams), dtype=np.int32)
    points = np.zeros((n_simulations, n_teams), dtype=np.int32)

    from src.table import PTS

    if hot_update and elo_ratings is not None:
        # --- Modo Hot: recalcula lambdas a cada jogo ---
        matches_info, strengths_indexed = _prepare_matches_hot(
            remaining_matches, team_strengths, team_idx,
        )

        desc = "Simulando (hot)"
        iterator = range(n_simulations)
        if show_progress:
            iterator = tqdm(iterator, desc=desc, unit="sim")

        for sim in iterator:
            data = base_data.copy()
            elo_sim = dict(elo_ratings)  # cópia dos ELOs

            pos = _simulate_season_hot(
                data, matches_info, strengths_indexed,
                league_avgs, elo_sim,
                hfa=HFA, k_sim=K_SIMULATION,
                elo_weight=ELO_LAMBDA_WEIGHT,
                rho=DIXON_COLES_RHO, max_goals=POISSON_MAX_GOALS,
            )

            positions[sim] = pos
            points[sim] = data[:, PTS]
    else:
        # --- Modo Estático: CDFs pré-computadas ---
        match_h_idx, match_a_idx, cdfs, score_h, score_a = _precompute_match_cdfs(
            remaining_matches, team_strengths, league_avgs, elo_ratings, team_idx,
        )

        desc = "Simulando"
        iterator = range(n_simulations)
        if show_progress:
            iterator = tqdm(iterator, desc=desc, unit="sim")

        for sim in iterator:
            data = base_data.copy()

            pos = _simulate_season_np(
                data, match_h_idx, match_a_idx, cdfs, score_h, score_a,
            )

            positions[sim] = pos
            points[sim] = data[:, PTS]

    return positions, points, team_names


# ---------------------------------------------------------------------------
# 4. Simulação com jogos fixados (What-if)
# ---------------------------------------------------------------------------

def run_monte_carlo_whatif(n_simulations, remaining_matches, current_table,
                          team_strengths, league_avgs, elo_ratings=None,
                          fixed_results=None, show_progress=True,
                          hot_update=HOT_UPDATE):
    """Monte Carlo com jogos fixados (cenário what-if).

    Args:
        fixed_results: dict {(mandante, visitante): (gols_h, gols_a)}
            Jogos cujo resultado é pré-definido.
        hot_update: Se True, usa Hot ELO + Hot Lambda.

    Returns:
        positions, points, team_names (mesma interface de run_monte_carlo)
    """
    if fixed_results is None:
        return run_monte_carlo(
            n_simulations, remaining_matches, current_table,
            team_strengths, league_avgs, elo_ratings, show_progress,
            hot_update=hot_update,
        )

    base_data, team_names, team_idx = table_to_arrays(current_table)
    n_teams = len(team_names)

    # Separar jogos fixados e livres
    fixed_matches = []
    free_matches = []
    for _, match in remaining_matches.iterrows():
        key = (match['mandante'], match['visitante'])
        if key in fixed_results:
            fixed_matches.append((
                team_idx[match['mandante']],
                team_idx[match['visitante']],
                match['mandante'],
                match['visitante'],
                fixed_results[key][0],
                fixed_results[key][1],
            ))
        else:
            free_matches.append(match)

    free_df = pd.DataFrame(free_matches) if free_matches else pd.DataFrame()

    positions = np.zeros((n_simulations, n_teams), dtype=np.int32)
    points_arr = np.zeros((n_simulations, n_teams), dtype=np.int32)

    from src.table import PTS

    if hot_update and elo_ratings is not None and len(free_df) > 0:
        # --- Hot mode ---
        matches_info_free, strengths_indexed = _prepare_matches_hot(
            free_df, team_strengths, team_idx,
        )

        iterator = range(n_simulations)
        if show_progress:
            iterator = tqdm(iterator, desc="Simulando what-if (hot)", unit="sim")

        for sim in iterator:
            data = base_data.copy()
            elo_sim = dict(elo_ratings)

            # Aplicar jogos fixados (e atualizar ELO com resultado fixo)
            for h_i, a_i, h_name, a_name, hg, ag in fixed_matches:
                update_table_np(data, h_i, a_i, hg, ag)
                # Hot ELO update para jogos fixados
                elo_h = elo_sim.get(h_name, 1500)
                elo_a = elo_sim.get(a_name, 1500)
                exp_h = 1.0 / (1.0 + 10.0 ** ((elo_a - (elo_h + HFA)) / 400.0))
                s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
                elo_sim[h_name] = elo_h + K_SIMULATION * (s_h - exp_h)
                elo_sim[a_name] = elo_a + K_SIMULATION * ((1.0 - s_h) - (1.0 - exp_h))

            # Simular jogos livres com hot update
            pos = _simulate_season_hot(
                data, matches_info_free, strengths_indexed,
                league_avgs, elo_sim,
                hfa=HFA, k_sim=K_SIMULATION,
                elo_weight=ELO_LAMBDA_WEIGHT,
                rho=DIXON_COLES_RHO, max_goals=POISSON_MAX_GOALS,
            )

            positions[sim] = pos
            points_arr[sim] = data[:, PTS]
    else:
        # --- Static mode ---
        if len(free_df) > 0:
            match_h_idx, match_a_idx, cdfs, score_h, score_a = _precompute_match_cdfs(
                free_df, team_strengths, league_avgs, elo_ratings, team_idx,
            )
        else:
            match_h_idx = np.array([], dtype=np.int32)

        iterator = range(n_simulations)
        if show_progress:
            iterator = tqdm(iterator, desc="Simulando (what-if)", unit="sim")

        for sim in iterator:
            data = base_data.copy()

            for h_i, a_i, _, _, hg, ag in fixed_matches:
                update_table_np(data, h_i, a_i, hg, ag)

            if len(match_h_idx) > 0:
                pos = _simulate_season_np(
                    data, match_h_idx, match_a_idx, cdfs, score_h, score_a,
                )
            else:
                pos = rank_teams_np(data)

            positions[sim] = pos
            points_arr[sim] = data[:, PTS]

    return positions, points_arr, team_names


# ---------------------------------------------------------------------------
# 5. Agregação de resultados
# ---------------------------------------------------------------------------

def aggregate_results(positions, points, team_names, n_simulations):
    """Agrega resultados das simulações em probabilidades por time."""
    n_teams = len(team_names)
    records = []

    for i, team in enumerate(team_names):
        pos_array = positions[:, i]
        pts_array = points[:, i]

        record = {
            'team': team,
            'p_titulo': np.mean(pos_array == 1),
            'p_libertadores': np.mean(pos_array <= 6),
            'p_sulamericana': np.mean((pos_array >= 7) & (pos_array <= 12)),
            'p_rebaixamento': np.mean(pos_array >= 17),
            'pts_mean': round(float(np.mean(pts_array)), 1),
            'pts_median': round(float(np.median(pts_array)), 0),
            'pts_min': int(np.min(pts_array)),
            'pts_max': int(np.max(pts_array)),
            'pts_p10': round(float(np.percentile(pts_array, 10)), 0),
            'pts_p90': round(float(np.percentile(pts_array, 90)), 0),
            'pos_mean': round(float(np.mean(pos_array)), 1),
        }

        # Distribuição de posições
        for p in range(1, n_teams + 1):
            record[f'pos_{p}'] = float(np.mean(pos_array == p))

        records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values('pos_mean').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 6. Backtest
# ---------------------------------------------------------------------------

def backtest(year, start_rounds, n_simulations=1000, db_path=DB_PATH,
             hot_update=HOT_UPDATE):
    """Backtest: simula temporada encerrada a partir de diferentes rodadas.

    Carrega temporada real, "congela" resultados até rodada N,
    simula o restante, e compara com resultado final real.
    """
    from src.elo import load_historical_matches, calculate_all_elos
    from src.poisson import load_serie_a_with_xg, calculate_team_strengths
    from src.load_data import _build_table_from_matches

    import sqlite3

    season_id = SERIE_A_IDS.get(year)
    if season_id is None:
        raise ValueError(f"Temporada {year} nao encontrada em SERIE_A_IDS")

    conn = sqlite3.connect(db_path)
    all_matches = pd.read_sql_query(f"""
        SELECT home_name, away_name, homeID, awayID,
               homeGoalCount, awayGoalCount,
               team_a_xg, team_b_xg,
               game_week, status, date_unix, competition_id
        FROM matches
        WHERE competition_id = {season_id}
          AND status = 'complete'
        ORDER BY game_week, date_unix
    """, conn)
    conn.close()

    for col in ['team_a_xg', 'team_b_xg']:
        all_matches[col] = pd.to_numeric(all_matches[col], errors='coerce')

    # Resultado final real
    final_table = _build_table_from_matches(season_id, db_path)
    real_champion = final_table.iloc[0]['team']
    real_relegated = set(final_table.tail(4)['team'].tolist())
    max_gw = all_matches['game_week'].max()

    # ELO
    elo_matches = load_historical_matches(db_path)
    elo_ratings, _ = calculate_all_elos(elo_matches)

    # Forças Poisson (temporada completa)
    matches_xg = load_serie_a_with_xg(db_path)
    target_matches = matches_xg[matches_xg['season_year'] == year]
    target_done = target_matches[target_matches['status'] == 'complete']
    team_strengths, league_avgs = calculate_team_strengths(target_done)

    backtest_results = []

    for start_round in start_rounds:
        if start_round > max_gw:
            continue

        print(f"  Backtest rodada {start_round}/{max_gw}...", end=" ", flush=True)

        completed = all_matches[all_matches['game_week'] <= start_round]
        partial_table = _build_partial_table(completed, final_table)

        future = all_matches[all_matches['game_week'] > start_round].copy()
        future = future.rename(columns={
            'home_name': 'mandante',
            'away_name': 'visitante',
            'homeID': 'mandante_id',
            'awayID': 'visitante_id',
            'game_week': 'rodada',
        })

        positions, pts, team_names = run_monte_carlo(
            n_simulations, future, partial_table,
            team_strengths, league_avgs, elo_ratings,
            show_progress=False, hot_update=hot_update,
        )

        agg = aggregate_results(positions, pts, team_names, n_simulations)

        champ_row = agg[agg['team'] == real_champion]
        p_champ = champ_row['p_titulo'].values[0] if len(champ_row) > 0 else 0

        p_rel_sum = 0
        for team in real_relegated:
            row = agg[agg['team'] == team]
            if len(row) > 0:
                p_rel_sum += row['p_rebaixamento'].values[0]
        p_rel_avg = p_rel_sum / len(real_relegated) if real_relegated else 0

        predicted_champ = agg.loc[agg['p_titulo'].idxmax(), 'team']

        backtest_results.append({
            'start_round': start_round,
            'remaining_games': len(future),
            'real_champion': real_champion,
            'p_real_champion_title': round(p_champ, 4),
            'predicted_champion': predicted_champ,
            'p_predicted_champion': round(float(agg['p_titulo'].max()), 4),
            'avg_p_relegated': round(p_rel_avg, 4),
            'champion_correct': predicted_champ == real_champion,
        })

        print(f"P(campeao real)={p_champ:.1%}, "
              f"Top pick={predicted_champ} ({agg['p_titulo'].max():.1%})")

    return pd.DataFrame(backtest_results)


def _build_partial_table(completed_matches, reference_table):
    """Constrói tabela parcial a partir de jogos realizados até rodada N."""
    all_teams = reference_table[['team', 'team_id']].copy()

    records = {}
    for _, row in all_teams.iterrows():
        records[row['team']] = {
            'team': row['team'], 'team_id': row['team_id'],
            'jogos': 0, 'vitorias': 0, 'empates': 0, 'derrotas': 0,
            'gols_pro': 0, 'gols_contra': 0, 'saldo': 0, 'pontos': 0,
        }

    for _, m in completed_matches.iterrows():
        hg, ag = int(m['homeGoalCount']), int(m['awayGoalCount'])
        home, away = m['home_name'], m['away_name']

        for team, gf, gc in [(home, hg, ag), (away, ag, hg)]:
            if team not in records:
                continue
            r = records[team]
            r['jogos'] += 1
            r['gols_pro'] += gf
            r['gols_contra'] += gc
            r['saldo'] += gf - gc
            if gf > gc:
                r['vitorias'] += 1
                r['pontos'] += 3
            elif gf == gc:
                r['empates'] += 1
                r['pontos'] += 1
            else:
                r['derrotas'] += 1

    df = pd.DataFrame(records.values())
    df = df.sort_values(
        by=['pontos', 'vitorias', 'saldo', 'gols_pro'],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    df['posicao'] = df.index + 1
    return df


# ---------------------------------------------------------------------------
# 7. Export
# ---------------------------------------------------------------------------

def export_results(results_df, year=TARGET_YEAR, data_dir=DATA_DIR):
    """Exporta resultados agregados para CSV."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "simulation_results.csv")
    results_df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# __main__: teste rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.elo import load_historical_matches, calculate_all_elos
    from src.poisson import load_serie_a_with_xg, calculate_team_strengths
    from src.load_data import (
        get_season_id, load_current_table, load_remaining_matches,
    )

    print("=" * 60)
    print(f"BRForecast — Fase 5: Simulação Monte Carlo")
    print("=" * 60)

    # --- 1. Carregar dados ---
    print(f"\n1. Carregando dados da temporada {TARGET_YEAR}...")
    season_id = get_season_id(TARGET_YEAR)
    table = load_current_table(season_id)
    remaining = load_remaining_matches(season_id)
    print(f"   {len(table)} times, {len(remaining)} jogos restantes")

    # --- 2. ELO ---
    print(f"\n2. Calculando ELOs...")
    elo_matches = load_historical_matches()
    elo_ratings, _ = calculate_all_elos(elo_matches)

    # --- 3. Forcas Poisson ---
    print(f"\n3. Calculando forcas Poisson...")
    matches_xg = load_serie_a_with_xg()
    target_m = matches_xg[matches_xg['season_year'] == TARGET_YEAR]
    target_done = target_m[target_m['status'] == 'complete']

    if len(target_done) < MIN_MATCHES_SEASON:
        expanded = matches_xg[matches_xg['season_year'] >= TARGET_YEAR - 1]
        team_strengths, league_avgs = calculate_team_strengths(expanded)
    else:
        team_strengths, league_avgs = calculate_team_strengths(target_done)

    # --- 4. Simulação ---
    n_sims = N_SIMULATIONS
    print(f"\n4. Rodando {n_sims} simulações...")

    positions, pts, team_names = run_monte_carlo(
        n_sims, remaining, table, team_strengths, league_avgs, elo_ratings,
    )

    # --- 5. Agregação ---
    print(f"\n5. Agregando resultados...")
    results = aggregate_results(positions, pts, team_names, n_sims)

    print(f"\n{'':3s}{'Time':25s} {'Titulo':>7s} {'Liberta':>8s} "
          f"{'Sulam':>7s} {'Rebx':>7s} {'Pts med':>8s} {'Pos med':>8s}")
    print(f"   {'-'*75}")
    for _, r in results.iterrows():
        print(f"   {r['team']:25s} {r['p_titulo']:>6.1%} {r['p_libertadores']:>7.1%} "
              f"{r['p_sulamericana']:>6.1%} {r['p_rebaixamento']:>6.1%} "
              f"{r['pts_mean']:>7.1f} {r['pos_mean']:>7.1f}")

    # --- 6. Export ---
    path = export_results(results)
    print(f"\n6. Exportado: {path}")

    # --- 7. Backtest ---
    print(f"\n7. Backtest (temporada 2025, 1000 sims por rodada)...")
    bt = backtest(2025, [10, 15, 20, 25, 30, 35], n_simulations=1000)
    print(f"\n   {'Rodada':>6s} {'Jogos rest':>10s} {'P(campeao real)':>15s} "
          f"{'Top pick':>20s} {'Acertou?':>8s}")
    print(f"   {'-'*65}")
    for _, r in bt.iterrows():
        print(f"   {r['start_round']:6d} {r['remaining_games']:10d} "
              f"{r['p_real_champion_title']:>14.1%} "
              f"{r['predicted_champion']:>20s} "
              f"{'SIM' if r['champion_correct'] else 'NAO':>8s}")

    print(f"\nFase 5 concluida.")
