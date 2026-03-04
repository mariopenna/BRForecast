"""BRForecast — Atualização de classificação durante simulação.

Duas interfaces:
  - DataFrame (API pública, usada fora da simulação)
  - NumPy arrays (interna, usada no loop Monte Carlo para performance)

Colunas do array numpy (índices):
  0=jogos, 1=vitorias, 2=empates, 3=derrotas,
  4=gols_pro, 5=gols_contra, 6=saldo, 7=pontos
"""

import numpy as np
import pandas as pd

# Índices das colunas no array numpy
J, V, E, D, GP, GC, SG, PTS = range(8)


def table_to_arrays(table_df):
    """Converte DataFrame de classificação para arrays numpy.

    Returns:
        data: ndarray (n_teams, 8) — stats por time
        team_names: list — nomes dos times (ordem fixa)
        team_idx: dict {name: index}
    """
    team_names = table_df['team'].tolist()
    team_idx = {t: i for i, t in enumerate(team_names)}
    n = len(team_names)

    data = np.zeros((n, 8), dtype=np.int32)
    for i, (_, row) in enumerate(table_df.iterrows()):
        data[i, J] = row['jogos']
        data[i, V] = row['vitorias']
        data[i, E] = row['empates']
        data[i, D] = row['derrotas']
        data[i, GP] = row['gols_pro']
        data[i, GC] = row['gols_contra']
        data[i, SG] = row['saldo']
        data[i, PTS] = row['pontos']

    return data, team_names, team_idx


def update_table_np(data, h_idx, a_idx, h_goals, a_goals):
    """Atualiza arrays numpy in-place com resultado de um jogo."""
    data[h_idx, J] += 1
    data[a_idx, J] += 1

    data[h_idx, GP] += h_goals
    data[h_idx, GC] += a_goals
    data[a_idx, GP] += a_goals
    data[a_idx, GC] += h_goals

    diff = h_goals - a_goals
    data[h_idx, SG] += diff
    data[a_idx, SG] -= diff

    if h_goals > a_goals:
        data[h_idx, V] += 1
        data[h_idx, PTS] += 3
        data[a_idx, D] += 1
    elif h_goals == a_goals:
        data[h_idx, E] += 1
        data[h_idx, PTS] += 1
        data[a_idx, E] += 1
        data[a_idx, PTS] += 1
    else:
        data[a_idx, V] += 1
        data[a_idx, PTS] += 3
        data[h_idx, D] += 1


def rank_teams_np(data):
    """Retorna array de posições (1-based) usando critérios do Brasileirão.

    Ordena por: pontos (desc), vitórias (desc), saldo (desc), gols pró (desc).
    """
    n = data.shape[0]
    # Criar chave de ordenação (negando para sort ascendente = desc)
    sort_key = np.lexsort((
        -data[:, GP],   # 4º critério: gols pró
        -data[:, SG],   # 3º critério: saldo
        -data[:, V],    # 2º critério: vitórias
        -data[:, PTS],  # 1º critério: pontos
    ))
    positions = np.empty(n, dtype=np.int32)
    positions[sort_key] = np.arange(1, n + 1)
    return positions


# ---------------------------------------------------------------------------
# API pública (DataFrame)
# ---------------------------------------------------------------------------

def update_table(table_df, home_team, away_team, home_goals, away_goals):
    """Atualiza a classificação com o resultado de um jogo (interface DataFrame)."""
    h_idx = table_df.index[table_df['team'] == home_team]
    a_idx = table_df.index[table_df['team'] == away_team]

    if len(h_idx) == 0 or len(a_idx) == 0:
        return

    h_idx = h_idx[0]
    a_idx = a_idx[0]

    table_df.at[h_idx, 'jogos'] += 1
    table_df.at[a_idx, 'jogos'] += 1

    table_df.at[h_idx, 'gols_pro'] += home_goals
    table_df.at[h_idx, 'gols_contra'] += away_goals
    table_df.at[a_idx, 'gols_pro'] += away_goals
    table_df.at[a_idx, 'gols_contra'] += home_goals

    table_df.at[h_idx, 'saldo'] += home_goals - away_goals
    table_df.at[a_idx, 'saldo'] += away_goals - home_goals

    if home_goals > away_goals:
        table_df.at[h_idx, 'vitorias'] += 1
        table_df.at[h_idx, 'pontos'] += 3
        table_df.at[a_idx, 'derrotas'] += 1
    elif home_goals == away_goals:
        table_df.at[h_idx, 'empates'] += 1
        table_df.at[h_idx, 'pontos'] += 1
        table_df.at[a_idx, 'empates'] += 1
        table_df.at[a_idx, 'pontos'] += 1
    else:
        table_df.at[a_idx, 'vitorias'] += 1
        table_df.at[a_idx, 'pontos'] += 3
        table_df.at[h_idx, 'derrotas'] += 1


def apply_tiebreakers(table_df):
    """Ordena a classificação pelos critérios do Brasileirão.

    Critérios: pontos > vitórias > saldo > gols marcados.
    """
    sorted_df = table_df.sort_values(
        by=['pontos', 'vitorias', 'saldo', 'gols_pro'],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    sorted_df['posicao'] = sorted_df.index + 1
    return sorted_df
