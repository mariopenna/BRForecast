"""BRForecast — Carga de dados da temporada alvo (2026).

Carrega a foto atual do campeonato: classificacao, jogos realizados,
jogos restantes. Faz merge com ratings ELO e forcas Poisson para
produzir um DataFrame completo por time.

Fontes:
  - league_tables: classificacao oficial do FootyStats
  - matches: jogos realizados e futuros da Serie A
  - elo_ratings: calculados em src/elo.py
  - team_strengths: calculados em src/poisson.py
"""

import sqlite3
import os
import pandas as pd
import numpy as np

from src.config import (
    DB_PATH, DATA_DIR, SERIE_A_IDS, TARGET_YEAR,
    HFA, ELO_LAMBDA_WEIGHT, MIN_MATCHES_SEASON,
)


# ---------------------------------------------------------------------------
# 1. Identificacao da temporada
# ---------------------------------------------------------------------------

def get_season_id(year=TARGET_YEAR):
    """Retorna o competition_id da Serie A para o ano dado."""
    sid = SERIE_A_IDS.get(year)
    if sid is None:
        available = sorted(SERIE_A_IDS.keys())
        raise ValueError(
            f"Serie A {year} nao encontrada. Disponiveis: {available}"
        )
    return sid


# ---------------------------------------------------------------------------
# 2. Classificacao atual
# ---------------------------------------------------------------------------

def load_current_table(season_id, db_path=DB_PATH):
    """Carrega classificacao atual da league_tables do FootyStats.

    Se nao houver dados em league_tables, constroi a partir de matches.

    Returns:
        DataFrame com colunas padronizadas:
        [team, team_id, jogos, vitorias, empates, derrotas,
         gols_pro, gols_contra, saldo, pontos, posicao]
    """
    conn = sqlite3.connect(db_path)

    # Tentar league_tables primeiro
    df = pd.read_sql_query(f"""
        SELECT cleanName AS team, team_id, matchesPlayed AS jogos,
               seasonWins_overall AS vitorias,
               seasonDraws_overall AS empates,
               seasonLosses_overall AS derrotas,
               seasonGoals AS gols_pro,
               seasonConceded AS gols_contra,
               seasonGoalDifference AS saldo,
               points AS pontos,
               position AS posicao,
               corrections
        FROM league_tables
        WHERE season_id = {season_id}
        ORDER BY posicao
    """, conn)
    conn.close()

    if len(df) > 0:
        # Aplicar correcoes de pontos (ex: punicoes)
        df['pontos'] = df['pontos'] + df['corrections'].fillna(0).astype(int)
        df = df.drop(columns=['corrections'])
        return df

    # Fallback: construir a partir de matches
    return _build_table_from_matches(season_id, db_path)


def _build_table_from_matches(season_id, db_path=DB_PATH):
    """Constroi classificacao a partir dos resultados de matches."""
    conn = sqlite3.connect(db_path)
    matches = pd.read_sql_query(f"""
        SELECT home_name, away_name, homeID, awayID,
               homeGoalCount, awayGoalCount
        FROM matches
        WHERE competition_id = {season_id}
          AND status = 'complete'
    """, conn)
    conn.close()

    if len(matches) == 0:
        return pd.DataFrame(columns=[
            'team', 'team_id', 'jogos', 'vitorias', 'empates', 'derrotas',
            'gols_pro', 'gols_contra', 'saldo', 'pontos', 'posicao',
        ])

    records = {}

    for _, m in matches.iterrows():
        hg, ag = int(m['homeGoalCount']), int(m['awayGoalCount'])

        for team, tid, gf, gc in [
            (m['home_name'], m['homeID'], hg, ag),
            (m['away_name'], m['awayID'], ag, hg),
        ]:
            if team not in records:
                records[team] = {
                    'team': team, 'team_id': tid,
                    'jogos': 0, 'vitorias': 0, 'empates': 0, 'derrotas': 0,
                    'gols_pro': 0, 'gols_contra': 0,
                }
            r = records[team]
            r['jogos'] += 1
            r['gols_pro'] += gf
            r['gols_contra'] += gc
            if gf > gc:
                r['vitorias'] += 1
            elif gf == gc:
                r['empates'] += 1
            else:
                r['derrotas'] += 1

    df = pd.DataFrame(records.values())
    df['saldo'] = df['gols_pro'] - df['gols_contra']
    df['pontos'] = df['vitorias'] * 3 + df['empates']
    df = df.sort_values(
        by=['pontos', 'vitorias', 'saldo', 'gols_pro'],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    df['posicao'] = df.index + 1

    return df


# ---------------------------------------------------------------------------
# 3. Jogos restantes (futuros)
# ---------------------------------------------------------------------------

def load_remaining_matches(season_id, db_path=DB_PATH):
    """Carrega jogos ainda nao realizados da temporada.

    Criterio: status != 'complete' (inclui 'incomplete' e 'suspended').

    Returns:
        DataFrame [rodada, mandante, visitante, mandante_id, visitante_id, date_unix, status]
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"""
        SELECT game_week AS rodada,
               home_name AS mandante,
               away_name AS visitante,
               homeID AS mandante_id,
               awayID AS visitante_id,
               date_unix,
               status
        FROM matches
        WHERE competition_id = {season_id}
          AND status != 'complete'
        ORDER BY game_week, date_unix
    """, conn)
    conn.close()

    return df


# ---------------------------------------------------------------------------
# 4. Jogos realizados
# ---------------------------------------------------------------------------

def load_completed_matches(season_id, db_path=DB_PATH):
    """Carrega jogos ja realizados da temporada com gols, xG, odds e stats extras.

    Returns:
        DataFrame com todas as colunas relevantes para analise individual:
        gols, xG, odds, chutes, posse, escanteios, ataques perigosos.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"""
        SELECT home_name, away_name,
               homeID, awayID,
               homeGoalCount, awayGoalCount,
               team_a_xg, team_b_xg,
               odds_ft_1, odds_ft_x, odds_ft_2,
               team_a_shots, team_b_shots,
               team_a_shotsOnTarget, team_b_shotsOnTarget,
               team_a_possession, team_b_possession,
               team_a_corners, team_b_corners,
               team_a_dangerous_attacks, team_b_dangerous_attacks,
               team_a_fouls, team_b_fouls,
               game_week, date_unix, status, id AS match_id
        FROM matches
        WHERE competition_id = {season_id}
          AND status = 'complete'
        ORDER BY game_week, date_unix
    """, conn)
    conn.close()

    for col in ['team_a_xg', 'team_b_xg']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ---------------------------------------------------------------------------
# 5. Merge: tabela + ELO + forcas Poisson
# ---------------------------------------------------------------------------

def build_team_stats(season_id, elo_ratings, team_strengths, league_avgs,
                     db_path=DB_PATH):
    """Monta DataFrame unificado: classificacao + ELO + poder ofensivo/defensivo.

    Args:
        season_id: ID da temporada alvo
        elo_ratings: dict {team_name: elo_rating} de src.elo
        team_strengths: DataFrame de src.poisson.calculate_team_strengths
        league_avgs: dict de src.poisson.calculate_team_strengths
        db_path: caminho para o banco SQLite

    Returns:
        DataFrame [team, team_id, jogos, pontos, posicao, vitorias, empates,
                   derrotas, gols_pro, gols_contra, saldo,
                   elo, attack, defense, avg_xg_for, avg_xg_against]
    """
    table = load_current_table(season_id, db_path)

    # Merge ELO
    elo_df = pd.DataFrame([
        {'team': team, 'elo': round(rating, 1)}
        for team, rating in elo_ratings.items()
    ])
    merged = table.merge(elo_df, on='team', how='left')

    # Merge team strengths
    strengths = team_strengths[['team', 'attack', 'defense', 'avg_for', 'avg_against']].copy()
    strengths = strengths.rename(columns={
        'avg_for': 'avg_xg_for',
        'avg_against': 'avg_xg_against',
    })
    merged = merged.merge(strengths, on='team', how='left')

    # Fallback para times sem ELO: usar 1500
    merged['elo'] = merged['elo'].fillna(1500.0)

    # Fallback para times sem strengths: usar media da liga (1.0)
    merged['attack'] = merged['attack'].fillna(1.0)
    merged['defense'] = merged['defense'].fillna(1.0)
    merged['avg_xg_for'] = merged['avg_xg_for'].fillna(league_avgs['avg_home_goals'])
    merged['avg_xg_against'] = merged['avg_xg_against'].fillna(league_avgs['avg_away_goals'])

    return merged


# ---------------------------------------------------------------------------
# 6. Export e __main__
# ---------------------------------------------------------------------------

def export_all(season_id, elo_ratings, team_strengths, league_avgs,
               db_path=DB_PATH, data_dir=DATA_DIR):
    """Exporta os 3 CSVs da temporada para data/.

    Returns:
        dict com DataFrames: {table, remaining, team_stats}
    """
    os.makedirs(data_dir, exist_ok=True)
    year = [y for y, sid in SERIE_A_IDS.items() if sid == season_id][0]

    # 1. Tabela atual
    table = load_current_table(season_id, db_path)
    table_path = os.path.join(data_dir, f"tabela_atual_{year}.csv")
    table.to_csv(table_path, index=False)

    # 2. Jogos restantes
    remaining = load_remaining_matches(season_id, db_path)
    remaining_path = os.path.join(data_dir, f"jogos_restantes_{year}.csv")
    remaining.to_csv(remaining_path, index=False)

    # 3. Team stats (merge)
    team_stats = build_team_stats(season_id, elo_ratings, team_strengths,
                                  league_avgs, db_path)
    stats_path = os.path.join(data_dir, f"team_stats_{year}.csv")
    team_stats.to_csv(stats_path, index=False)

    return {
        'table': table,
        'remaining': remaining,
        'team_stats': team_stats,
    }


if __name__ == "__main__":
    from src.elo import load_historical_matches, calculate_all_elos
    from src.poisson import load_serie_a_with_xg, calculate_team_strengths

    print("=" * 60)
    print(f"BRForecast — Fase 4: Dados da temporada {TARGET_YEAR}")
    print("=" * 60)

    season_id = get_season_id(TARGET_YEAR)
    print(f"\nSeason ID: {season_id}")

    # --- 1. Classificacao atual ---
    print(f"\n1. Carregando classificacao atual...")
    table = load_current_table(season_id)
    print(f"   {len(table)} times na tabela")
    print(f"\n   {'Pos':>3s} {'Time':25s} {'J':>3s} {'V':>3s} {'E':>3s} {'D':>3s} "
          f"{'GP':>3s} {'GC':>3s} {'SG':>4s} {'Pts':>3s}")
    print(f"   {'-'*60}")
    for _, r in table.iterrows():
        print(f"   {r['posicao']:3.0f} {r['team']:25s} {r['jogos']:3d} "
              f"{r['vitorias']:3d} {r['empates']:3d} {r['derrotas']:3d} "
              f"{r['gols_pro']:3d} {r['gols_contra']:3d} {r['saldo']:+4d} "
              f"{r['pontos']:3d}")

    # --- 2. Jogos restantes ---
    print(f"\n2. Carregando jogos restantes...")
    remaining = load_remaining_matches(season_id)
    print(f"   {len(remaining)} jogos restantes")
    gw_counts = remaining.groupby('rodada').size()
    print(f"   Rodadas: {remaining['rodada'].min()} a {remaining['rodada'].max()}")
    print(f"   Jogos por rodada (primeiras 3):")
    for gw in sorted(gw_counts.index)[:3]:
        print(f"     GW {gw:2d}: {gw_counts[gw]} jogos")

    suspended = remaining[remaining['status'] == 'suspended']
    if len(suspended) > 0:
        print(f"\n   Jogos suspensos ({len(suspended)}):")
        for _, s in suspended.iterrows():
            print(f"     GW{s['rodada']}: {s['mandante']} vs {s['visitante']}")

    # --- 3. Jogos realizados ---
    print(f"\n3. Carregando jogos realizados...")
    completed = load_completed_matches(season_id)
    n_xg = completed.dropna(subset=['team_a_xg', 'team_b_xg']).shape[0]
    print(f"   {len(completed)} jogos realizados, {n_xg} com xG")

    # --- 4. ELO ---
    print(f"\n4. Calculando ELOs...")
    elo_matches = load_historical_matches()
    elo_ratings, _ = calculate_all_elos(elo_matches)
    print(f"   {len(elo_ratings)} times com ELO")

    # --- 5. Forcas Poisson ---
    print(f"\n5. Calculando forcas ofensivas/defensivas...")
    matches_xg = load_serie_a_with_xg()
    target_matches = matches_xg[matches_xg['season_year'] == TARGET_YEAR]
    target_done = target_matches[target_matches['status'] == 'complete']

    if len(target_done) < MIN_MATCHES_SEASON:
        print(f"   Apenas {len(target_done)} jogos em {TARGET_YEAR} "
              f"(minimo={MIN_MATCHES_SEASON}). Incluindo {TARGET_YEAR - 1}.")
        expanded = matches_xg[matches_xg['season_year'] >= TARGET_YEAR - 1]
        team_strengths, league_avgs = calculate_team_strengths(expanded)
    else:
        team_strengths, league_avgs = calculate_team_strengths(target_done)
    print(f"   {len(team_strengths)} times com forcas calculadas")

    # --- 6. Merge e export ---
    print(f"\n6. Fazendo merge e exportando CSVs...")
    result = export_all(season_id, elo_ratings, team_strengths, league_avgs)

    ts = result['team_stats']
    print(f"\n   === Team Stats Completo ({len(ts)} times) ===")
    print(f"   {'Pos':>3s} {'Time':25s} {'Pts':>3s} {'ELO':>6s} "
          f"{'Atk':>6s} {'Def':>6s} {'xG/j':>6s} {'xGA/j':>6s}")
    print(f"   {'-'*70}")
    for _, r in ts.iterrows():
        print(f"   {r['posicao']:3.0f} {r['team']:25s} {r['pontos']:3d} "
              f"{r['elo']:6.1f} {r['attack']:6.3f} {r['defense']:6.3f} "
              f"{r['avg_xg_for']:6.3f} {r['avg_xg_against']:6.3f}")

    # --- 7. Checagens ---
    print(f"\n7. Checagens finais...")
    n_nan_elo = ts['elo'].isna().sum()
    n_nan_atk = ts['attack'].isna().sum()
    n_nan_def = ts['defense'].isna().sum()
    print(f"   NaN em ELO: {n_nan_elo}")
    print(f"   NaN em attack: {n_nan_atk}")
    print(f"   NaN em defense: {n_nan_def}")

    total_matches = len(completed) + len(remaining)
    print(f"   Total jogos (realizados + restantes): {total_matches}")
    expected_total = 380  # 20 times × 38 rodadas / 2
    print(f"   Esperado: {expected_total}")
    if total_matches == expected_total:
        print(f"   OK: total bate!")
    else:
        print(f"   AVISO: diferenca de {total_matches - expected_total} jogos")

    print(f"\n   CSVs exportados em data/:")
    for f in os.listdir(DATA_DIR):
        if f.endswith('.csv'):
            size = os.path.getsize(os.path.join(DATA_DIR, f))
            print(f"   - {f} ({size:,} bytes)")

    print(f"\nFase 4 concluida.")
