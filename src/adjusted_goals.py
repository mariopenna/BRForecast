"""BRForecast — Adjusted Goals (gols ajustados).

Nem todo gol tem o mesmo peso informativo sobre a forca de um time.
Gols tardios com placar ja definido ou gols contra 10 jogadores sao
menos informativos. Este modulo calcula gols ajustados por partida.

Regras de ajuste:
  1. Gols tardios com placar definido valem menos:
     - min >= 80 e diff >= 2 antes do gol -> peso 0.5
     - min >= 85 e diff >= 1 antes do gol -> peso 0.75
  2. Gols com vantagem numerica (adversario expulso) valem menos:
     - Se adversario tinha vermelho antes do gol -> peso 0.8
  3. Peso final = peso_tardio * peso_vantagem_numerica

Fonte de dados:
  - match_details.team_a_goal_details / team_b_goal_details (JSON)
  - match_details.team_a_card_details / team_b_card_details (JSON)
"""

import json
import re
import sqlite3
import pandas as pd
import numpy as np

from src.config import DB_PATH, SERIE_A_IDS, ELO_WINDOW_START


# ---------------------------------------------------------------------------
# 1. Parsing de minutos
# ---------------------------------------------------------------------------

def _parse_minute(time_str):
    """Converte string de minuto para float.

    Exemplos:
        "45"    -> 45.0
        "90+2"  -> 92.0
        "86'"   -> 86.0
        "45+1'" -> 46.0
    """
    if not time_str:
        return None
    s = str(time_str).strip().rstrip("'")
    match = re.match(r"(\d+)\+(\d+)", s)
    if match:
        return float(match.group(1)) + float(match.group(2))
    try:
        return float(s)
    except ValueError:
        return None


def _parse_json_safe(text):
    """Parse JSON com fallback para lista vazia."""
    if not text or text == 'null':
        return []
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# 2. Calculo de peso de cada gol
# ---------------------------------------------------------------------------

def _goal_weight_late(minute, goal_diff_before):
    """Peso por gol tardio com placar definido.

    Args:
        minute: minuto do gol (float)
        goal_diff_before: diferenca de gols ANTES deste gol
                          (positivo = time que marcou ja liderava)
    """
    if minute is None:
        return 1.0
    if minute >= 80 and abs(goal_diff_before) >= 2:
        return 0.5
    if minute >= 85 and abs(goal_diff_before) >= 1:
        return 0.75
    return 1.0


def _goal_weight_red_card(minute, opponent_red_minutes):
    """Peso por vantagem numerica.

    Args:
        minute: minuto do gol
        opponent_red_minutes: lista de minutos em que o adversario
                              recebeu vermelho (direto ou segundo amarelo)
    """
    if minute is None or not opponent_red_minutes:
        return 1.0
    # Se algum vermelho do adversario foi antes deste gol
    for red_min in opponent_red_minutes:
        if red_min is not None and red_min < minute:
            return 0.8
    return 1.0


def compute_adjusted_goals_match(goal_details_scoring, goal_details_conceding,
                                 card_details_opponent):
    """Calcula gols ajustados para UM time em UMA partida.

    Args:
        goal_details_scoring: lista de dicts com gols do time que marcou
                              (cada dict tem 'time')
        goal_details_conceding: lista de dicts com gols do adversario
                                (para calcular goal_diff antes de cada gol)
        card_details_opponent: lista de dicts com cartoes do adversario
                               (para detectar vermelhos)

    Returns:
        float: soma dos pesos dos gols (adjusted goals)
    """
    # Extrair minutos dos vermelhos do adversario
    opponent_reds = []
    for card in card_details_opponent:
        card_type = card.get('card_type', '')
        if card_type in ('Red', 'Second Yellow'):
            m = _parse_minute(card.get('time'))
            if m is not None:
                opponent_reds.append(m)

    # Extrair todos os gols de ambos os times com minutos, para reconstruir
    # o placar parcial antes de cada gol
    all_goals = []
    for g in goal_details_scoring:
        m = _parse_minute(g.get('time'))
        all_goals.append(('scoring', m))
    for g in goal_details_conceding:
        m = _parse_minute(g.get('time'))
        all_goals.append(('conceding', m))

    # Ordenar por minuto (gols sem minuto vao para o fim)
    all_goals.sort(key=lambda x: x[1] if x[1] is not None else 999)

    # Calcular peso de cada gol do time que marcou
    total_weight = 0.0
    score_for = 0
    score_against = 0

    for side, minute in all_goals:
        if side == 'scoring':
            diff_before = score_for - score_against
            w_late = _goal_weight_late(minute, diff_before)
            w_red = _goal_weight_red_card(minute, opponent_reds)
            total_weight += w_late * w_red
            score_for += 1
        else:
            score_against += 1

    return total_weight


# ---------------------------------------------------------------------------
# 3. Carga de dados do banco
# ---------------------------------------------------------------------------

def load_adjusted_goals(db_path=DB_PATH, start_year=ELO_WINDOW_START):
    """Carrega e computa adjusted goals para todas as partidas da Serie A.

    Faz JOIN entre matches e match_details para obter goal/card details.

    Returns:
        DataFrame com colunas:
        [match_id, home_name, away_name, season_year, date_unix,
         homeGoalCount, awayGoalCount, adj_goals_home, adj_goals_away]
    """
    season_filter = {y: sid for y, sid in SERIE_A_IDS.items() if y >= start_year}
    season_ids = tuple(season_filter.values())

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"""
        SELECT m.id AS match_id,
               m.home_name, m.away_name,
               m.homeGoalCount, m.awayGoalCount,
               m.date_unix, m.status,
               l.year AS season_year,
               md.team_a_goal_details,
               md.team_b_goal_details,
               md.team_a_card_details,
               md.team_b_card_details
        FROM matches m
        JOIN leagues l ON m.competition_id = l.id
        LEFT JOIN match_details md ON m.id = md.id
        WHERE m.competition_id IN {season_ids}
          AND m.status = 'complete'
        ORDER BY m.date_unix
    """, conn)
    conn.close()

    adj_home = []
    adj_away = []

    for _, row in df.iterrows():
        goals_a_details = _parse_json_safe(row['team_a_goal_details'])
        goals_b_details = _parse_json_safe(row['team_b_goal_details'])
        cards_a_details = _parse_json_safe(row['team_a_card_details'])
        cards_b_details = _parse_json_safe(row['team_b_card_details'])

        # Adjusted goals do mandante: seus gols, gols do visitante, cartoes do visitante
        adj_h = compute_adjusted_goals_match(
            goals_a_details, goals_b_details, cards_b_details,
        )
        # Adjusted goals do visitante: seus gols, gols do mandante, cartoes do mandante
        adj_a = compute_adjusted_goals_match(
            goals_b_details, goals_a_details, cards_a_details,
        )

        adj_home.append(adj_h)
        adj_away.append(adj_a)

    df['adj_goals_home'] = adj_home
    df['adj_goals_away'] = adj_away

    return df[['match_id', 'home_name', 'away_name', 'season_year', 'date_unix',
               'homeGoalCount', 'awayGoalCount', 'adj_goals_home', 'adj_goals_away']]


# ---------------------------------------------------------------------------
# __main__: diagnostico
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("BRForecast — Adjusted Goals")
    print("=" * 60)

    print("\nCarregando e calculando adjusted goals...")
    adj_df = load_adjusted_goals()
    print(f"  {len(adj_df)} partidas processadas")

    # Estatisticas
    adj_df['raw_total'] = adj_df['homeGoalCount'] + adj_df['awayGoalCount']
    adj_df['adj_total'] = adj_df['adj_goals_home'] + adj_df['adj_goals_away']
    adj_df['discount'] = adj_df['raw_total'] - adj_df['adj_total']

    print(f"\n  Media gols reais por jogo:    {adj_df['raw_total'].mean():.3f}")
    print(f"  Media gols ajustados por jogo: {adj_df['adj_total'].mean():.3f}")
    print(f"  Desconto medio por jogo:       {adj_df['discount'].mean():.3f}")
    print(f"  % jogos com desconto:          {(adj_df['discount'] > 0).mean():.1%}")

    # Top 10 maiores descontos
    top_disc = adj_df.nlargest(10, 'discount')
    print(f"\n  === Top 10 maiores descontos ===")
    print(f"  {'Jogo':45s} {'Placar':>7s} {'Adj':>10s} {'Desc':>5s}")
    print(f"  {'-'*70}")
    for _, r in top_disc.iterrows():
        jogo = f"{r['home_name']} vs {r['away_name']}"
        placar = f"{int(r['homeGoalCount'])}-{int(r['awayGoalCount'])}"
        adj = f"{r['adj_goals_home']:.1f}-{r['adj_goals_away']:.1f}"
        print(f"  {jogo:45s} {placar:>7s} {adj:>10s} {r['discount']:>5.2f}")

    # Por temporada
    print(f"\n  === Desconto medio por temporada ===")
    by_year = adj_df.groupby('season_year').agg(
        n_matches=('match_id', 'count'),
        avg_raw=('raw_total', 'mean'),
        avg_adj=('adj_total', 'mean'),
        avg_discount=('discount', 'mean'),
    )
    for year, r in by_year.iterrows():
        print(f"  {year}: {r['avg_raw']:.2f} -> {r['avg_adj']:.2f} "
              f"(desconto {r['avg_discount']:.3f}, n={int(r['n_matches'])})")
