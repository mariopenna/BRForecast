"""BRForecast — Analise individual de jogos (merecimento via xG).

Para cada jogo ja realizado da Serie A, classifica o resultado como
"Merecido", "Parcialmente injusto" ou "Muito injusto" com base na
diferenca entre xG e gols reais. Gera ficha resumo por partida.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from src.poisson import (
    calculate_lambdas,
    score_probabilities,
    odds_to_probs,
)


# ---------------------------------------------------------------------------
# 1. Classificacao de merecimento
# ---------------------------------------------------------------------------

def _determine_result(goals_h, goals_a, margin=0.0):
    """Determina resultado: 'H' (home), 'D' (draw), 'A' (away).

    Se margin > 0, usa como zona de empate (ex: |diff| <= margin => D).
    """
    diff = goals_h - goals_a
    if abs(diff) <= margin:
        return 'D'
    return 'H' if diff > 0 else 'A'


def classify_merit(goals_h, goals_a, xg_h, xg_a):
    """Classifica o merecimento do resultado de uma partida.

    Compara o resultado real (gols) com o resultado "esperado" (xG).
    Usa margem de 0.3 para empate no xG (ex: xG 1.2 vs 1.0 = empate).

    Args:
        goals_h: gols do mandante
        goals_a: gols do visitante
        xg_h: xG do mandante
        xg_a: xG do visitante

    Returns:
        dict com:
            verdict: "Merecido", "Parcialmente injusto", "Muito injusto"
            result_real: 'H'/'D'/'A'
            result_xg: 'H'/'D'/'A'
            luck_home: gols_h - xg_h (positivo = sorte)
            luck_away: gols_a - xg_a
            xg_diff: xg_h - xg_a (positivo = mandante mereceu mais)
            goals_diff: goals_h - goals_a
    """
    if pd.isna(xg_h) or pd.isna(xg_a):
        return {
            'verdict': 'Sem xG',
            'result_real': _determine_result(goals_h, goals_a),
            'result_xg': None,
            'luck_home': None,
            'luck_away': None,
            'xg_diff': None,
            'goals_diff': goals_h - goals_a,
        }

    result_real = _determine_result(goals_h, goals_a)
    result_xg = _determine_result(xg_h, xg_a, margin=0.3)

    luck_home = goals_h - xg_h
    luck_away = goals_a - xg_a

    # Diferenca entre sorte dos dois lados
    # |luck_diff| grande = resultado muito distorcido pela sorte
    luck_diff = abs((goals_h - goals_a) - (xg_h - xg_a))

    if result_real == result_xg:
        verdict = 'Merecido'
    elif luck_diff > 1.5:
        verdict = 'Muito injusto'
    else:
        verdict = 'Parcialmente injusto'

    return {
        'verdict': verdict,
        'result_real': result_real,
        'result_xg': result_xg,
        'luck_home': round(luck_home, 2),
        'luck_away': round(luck_away, 2),
        'xg_diff': round(xg_h - xg_a, 2),
        'goals_diff': goals_h - goals_a,
    }


# ---------------------------------------------------------------------------
# 2. Ficha de partida
# ---------------------------------------------------------------------------

def build_match_card(match_row, team_strengths, league_avgs,
                     elo_ratings=None):
    """Constroi ficha completa de analise para uma partida.

    Args:
        match_row: Series/dict com dados da partida (de load_completed_matches)
        team_strengths: DataFrame de calculate_team_strengths
        league_avgs: dict de calculate_team_strengths
        elo_ratings: dict {team: elo} opcional

    Returns:
        dict com todas as metricas da ficha
    """
    home = match_row['home_name']
    away = match_row['away_name']
    goals_h = int(match_row['homeGoalCount'])
    goals_a = int(match_row['awayGoalCount'])
    xg_h = match_row.get('team_a_xg')
    xg_a = match_row.get('team_b_xg')

    # Converter xG para float se possivel
    xg_h = float(xg_h) if pd.notna(xg_h) else None
    xg_a = float(xg_a) if pd.notna(xg_a) else None

    # --- Merecimento ---
    merit = classify_merit(goals_h, goals_a, xg_h, xg_a)

    # --- Probabilidades do modelo Poisson ---
    lam_h, lam_a = calculate_lambdas(
        home, away, team_strengths, league_avgs, elo_ratings=elo_ratings,
    )
    probs = score_probabilities(lam_h, lam_a)

    # Probabilidade do placar exato que ocorreu
    matrix = probs['matrix']
    if goals_h < matrix.shape[0] and goals_a < matrix.shape[1]:
        p_exact_score = float(matrix[goals_h, goals_a])
    else:
        p_exact_score = 0.0

    # --- Odds (se disponiveis) ---
    odds_h = match_row.get('odds_ft_1')
    odds_d = match_row.get('odds_ft_x')
    odds_a = match_row.get('odds_ft_2')

    has_odds = (pd.notna(odds_h) and pd.notna(odds_d) and pd.notna(odds_a)
                and odds_h > 0 and odds_d > 0 and odds_a > 0)

    if has_odds:
        odds_p_h, odds_p_d, odds_p_a = odds_to_probs(
            float(odds_h), float(odds_d), float(odds_a)
        )
    else:
        odds_p_h = odds_p_d = odds_p_a = None

    # --- Resultado ---
    if goals_h > goals_a:
        result = 'H'
        result_text = f'{home} venceu'
    elif goals_h == goals_a:
        result = 'D'
        result_text = 'Empate'
    else:
        result = 'A'
        result_text = f'{away} venceu'

    # --- Data ---
    date_unix = match_row.get('date_unix')
    if pd.notna(date_unix):
        date_str = datetime.fromtimestamp(int(date_unix), tz=timezone.utc).strftime('%d/%m/%Y')
    else:
        date_str = None

    # --- Ficha ---
    card = {
        # Identificacao
        'home': home,
        'away': away,
        'game_week': match_row.get('game_week'),
        'date': date_str,
        'date_unix': date_unix,
        'match_id': match_row.get('match_id'),

        # Placar
        'goals_home': goals_h,
        'goals_away': goals_a,
        'result': result,
        'result_text': result_text,

        # xG
        'xg_home': xg_h,
        'xg_away': xg_a,
        'has_xg': xg_h is not None and xg_a is not None,

        # Merecimento
        'verdict': merit['verdict'],
        'result_xg': merit['result_xg'],
        'luck_home': merit['luck_home'],
        'luck_away': merit['luck_away'],
        'xg_diff': merit['xg_diff'],
        'goals_diff': merit['goals_diff'],

        # Modelo Poisson
        'lambda_home': round(lam_h, 3),
        'lambda_away': round(lam_a, 3),
        'p_home_win': round(probs['home_win'], 4),
        'p_draw': round(probs['draw'], 4),
        'p_away_win': round(probs['away_win'], 4),
        'p_exact_score': round(p_exact_score, 4),

        # Odds
        'has_odds': has_odds,
        'odds_home': float(odds_h) if has_odds else None,
        'odds_draw': float(odds_d) if has_odds else None,
        'odds_away': float(odds_a) if has_odds else None,
        'odds_p_home': round(odds_p_h, 4) if odds_p_h is not None else None,
        'odds_p_draw': round(odds_p_d, 4) if odds_p_d is not None else None,
        'odds_p_away': round(odds_p_a, 4) if odds_p_a is not None else None,
    }

    # --- Stats extras (se disponiveis) ---
    extra_stats = {
        'shots_home': 'team_a_shots',
        'shots_away': 'team_b_shots',
        'shots_on_target_home': 'team_a_shotsOnTarget',
        'shots_on_target_away': 'team_b_shotsOnTarget',
        'possession_home': 'team_a_possession',
        'possession_away': 'team_b_possession',
        'corners_home': 'team_a_corners',
        'corners_away': 'team_b_corners',
        'dangerous_attacks_home': 'team_a_dangerous_attacks',
        'dangerous_attacks_away': 'team_b_dangerous_attacks',
        'fouls_home': 'team_a_fouls',
        'fouls_away': 'team_b_fouls',
    }

    has_extra = False
    for card_key, db_col in extra_stats.items():
        val = match_row.get(db_col)
        if pd.notna(val) and val != 0:
            card[card_key] = int(val)
            has_extra = True
        else:
            card[card_key] = None

    card['has_extra_stats'] = has_extra

    return card


# ---------------------------------------------------------------------------
# 3. Analise de todos os jogos
# ---------------------------------------------------------------------------

def analyze_all_matches(completed_matches, team_strengths, league_avgs,
                        elo_ratings=None):
    """Aplica build_match_card para cada jogo realizado.

    Args:
        completed_matches: DataFrame de load_completed_matches
        team_strengths: DataFrame de calculate_team_strengths
        league_avgs: dict de calculate_team_strengths
        elo_ratings: dict {team: elo} opcional

    Returns:
        DataFrame com uma linha por jogo e todas as metricas da ficha.
    """
    cards = []
    for _, match_row in completed_matches.iterrows():
        card = build_match_card(
            match_row, team_strengths, league_avgs,
            elo_ratings=elo_ratings,
        )
        cards.append(card)

    if not cards:
        return pd.DataFrame()

    return pd.DataFrame(cards)


# ---------------------------------------------------------------------------
# 4. Resultados mais imerecidos
# ---------------------------------------------------------------------------

def get_most_undeserved(match_cards_df, top_n=10):
    """Retorna os N jogos com maior discrepancia entre gols e xG.

    Ordena por |goals_diff - xg_diff| descendente (maior "sorte/azar").

    Args:
        match_cards_df: DataFrame retornado por analyze_all_matches
        top_n: numero de jogos a retornar

    Returns:
        DataFrame com os jogos mais imerecidos
    """
    df = match_cards_df[match_cards_df['has_xg']].copy()

    if df.empty:
        return df

    # |goals_diff - xg_diff| mede a distorcao total do resultado
    df['luck_score'] = (df['goals_diff'] - df['xg_diff']).abs()

    return (
        df.sort_values('luck_score', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 5. Sumario de merecimento da temporada
# ---------------------------------------------------------------------------

def merit_summary(match_cards_df):
    """Resume a distribuicao de merecimento da temporada.

    Returns:
        dict com contagens e percentuais por veredicto
    """
    with_xg = match_cards_df[match_cards_df['has_xg']]

    if with_xg.empty:
        return {'total': 0, 'with_xg': 0}

    counts = with_xg['verdict'].value_counts()
    total = len(with_xg)

    return {
        'total': len(match_cards_df),
        'with_xg': total,
        'merecido': int(counts.get('Merecido', 0)),
        'merecido_pct': counts.get('Merecido', 0) / total,
        'parcialmente': int(counts.get('Parcialmente injusto', 0)),
        'parcialmente_pct': counts.get('Parcialmente injusto', 0) / total,
        'muito': int(counts.get('Muito injusto', 0)),
        'muito_pct': counts.get('Muito injusto', 0) / total,
        'avg_luck_home': with_xg['luck_home'].mean(),
        'avg_luck_away': with_xg['luck_away'].mean(),
    }


# ---------------------------------------------------------------------------
# __main__: roda analise completa da Fase 7
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.config import TARGET_YEAR, MIN_MATCHES_SEASON
    from src.elo import load_historical_matches, calculate_all_elos
    from src.poisson import load_serie_a_with_xg, calculate_team_strengths
    from src.load_data import get_season_id, load_completed_matches

    print("=" * 60)
    print(f"BRForecast — Fase 7: Visao individual de jogos")
    print("=" * 60)

    # --- 1. Carregar dados base ---
    print("\n1. Carregando dados base...")
    season_id = get_season_id(TARGET_YEAR)
    print(f"   Season ID: {season_id}")

    # ELO
    elo_matches = load_historical_matches()
    elo_ratings, _ = calculate_all_elos(elo_matches)
    print(f"   {len(elo_ratings)} times com ELO")

    # Forcas Poisson
    matches_xg = load_serie_a_with_xg()
    target = matches_xg[matches_xg['season_year'] == TARGET_YEAR]
    target_done = target[target['status'] == 'complete']

    if len(target_done) < MIN_MATCHES_SEASON:
        print(f"   {len(target_done)} jogos em {TARGET_YEAR} "
              f"(minimo={MIN_MATCHES_SEASON}). Incluindo {TARGET_YEAR - 1}.")
        expanded = matches_xg[matches_xg['season_year'] >= TARGET_YEAR - 1]
        team_strengths, league_avgs = calculate_team_strengths(expanded)
    else:
        team_strengths, league_avgs = calculate_team_strengths(target_done)
    print(f"   {len(team_strengths)} times com forcas calculadas")

    # Jogos realizados
    completed = load_completed_matches(season_id)
    n_xg = completed.dropna(subset=['team_a_xg', 'team_b_xg']).shape[0]
    print(f"   {len(completed)} jogos realizados, {n_xg} com xG")

    # --- 2. Gerar fichas para todos os jogos ---
    print(f"\n2. Gerando fichas de analise...")
    cards_df = analyze_all_matches(
        completed, team_strengths, league_avgs, elo_ratings=elo_ratings,
    )
    print(f"   {len(cards_df)} fichas geradas")

    # --- 3. Sumario de merecimento ---
    print(f"\n3. Sumario de merecimento:")
    summary = merit_summary(cards_df)
    print(f"   Total de jogos:         {summary['total']}")
    print(f"   Jogos com xG:           {summary['with_xg']}")
    if summary['with_xg'] > 0:
        print(f"   Merecido:               {summary['merecido']:3d} "
              f"({summary['merecido_pct']:.1%})")
        print(f"   Parcialmente injusto: {summary['parcialmente']:3d} "
              f"({summary['parcialmente_pct']:.1%})")
        print(f"   Muito injusto:        {summary['muito']:3d} "
              f"({summary['muito_pct']:.1%})")
        print(f"   Sorte media mandante:   {summary['avg_luck_home']:+.3f}")
        print(f"   Sorte media visitante:  {summary['avg_luck_away']:+.3f}")

    # --- 4. Top 10 mais imerecidos ---
    print(f"\n4. Top 10 resultados mais imerecidos:")
    top = get_most_undeserved(cards_df, top_n=10)

    if len(top) == 0:
        print("   Nenhum jogo com xG disponivel.")
    else:
        print(f"   {'GW':>3s}  {'Jogo':45s} {'Placar':>7s} {'xG':>10s} "
              f"{'Veredicto':>22s} {'Sorte':>6s}")
        print(f"   {'-'*98}")
        for _, r in top.iterrows():
            placar = f"{r['goals_home']}-{r['goals_away']}"
            xg_str = f"{r['xg_home']:.1f}-{r['xg_away']:.1f}"
            jogo = f"{r['home']} vs {r['away']}"
            print(f"   {r['game_week']:3.0f}  {jogo:45s} {placar:>7s} "
                  f"{xg_str:>10s} {r['verdict']:>22s} "
                  f"{r['luck_score']:>5.2f}")

    # --- 5. Exemplo de ficha detalhada ---
    if len(top) > 0:
        print(f"\n5. Ficha detalhada do jogo mais imerecido:")
        r = top.iloc[0]
        print(f"   {r['home']} {r['goals_home']}-{r['goals_away']} {r['away']}")
        print(f"   Rodada {r['game_week']} | {r['date']}")
        print(f"   Resultado: {r['result_text']}")
        print(f"   Veredicto: {r['verdict']}")
        print(f"\n   xG:")
        print(f"     {r['home']:25s} {r['xg_home']:.2f} (gols: {r['goals_home']}, "
              f"sorte: {r['luck_home']:+.2f})")
        print(f"     {r['away']:25s} {r['xg_away']:.2f} (gols: {r['goals_away']}, "
              f"sorte: {r['luck_away']:+.2f})")
        print(f"     xG diff: {r['xg_diff']:+.2f} | "
              f"Resultado xG: {r['result_xg']}")
        print(f"\n   Modelo Poisson:")
        print(f"     P(vitoria {r['home']}): {r['p_home_win']:.1%}")
        print(f"     P(empate):             {r['p_draw']:.1%}")
        print(f"     P(vitoria {r['away']}): {r['p_away_win']:.1%}")
        print(f"     P(placar exato {r['goals_home']}-{r['goals_away']}): "
              f"{r['p_exact_score']:.1%}")

        if r['has_odds']:
            print(f"\n   Odds:")
            print(f"     Casa:    {r['odds_home']:.2f} -> {r['odds_p_home']:.1%}")
            print(f"     Empate:  {r['odds_draw']:.2f} -> {r['odds_p_draw']:.1%}")
            print(f"     Fora:    {r['odds_away']:.2f} -> {r['odds_p_away']:.1%}")

        if r['has_extra_stats']:
            print(f"\n   Stats extras:")
            if pd.notna(r['shots_home']):
                print(f"     Chutes:       {int(r['shots_home']):3d} vs {int(r['shots_away']):3d}")
            if pd.notna(r['shots_on_target_home']):
                print(f"     No alvo:      {int(r['shots_on_target_home']):3d} vs "
                      f"{int(r['shots_on_target_away']):3d}")
            if pd.notna(r['possession_home']):
                print(f"     Posse:        {int(r['possession_home']):3d}% vs "
                      f"{int(r['possession_away']):3d}%")
            if pd.notna(r['corners_home']):
                print(f"     Escanteios:   {int(r['corners_home']):3d} vs "
                      f"{int(r['corners_away']):3d}")
            if pd.notna(r['dangerous_attacks_home']):
                print(f"     Ataq. perig.: {int(r['dangerous_attacks_home']):3d} vs "
                      f"{int(r['dangerous_attacks_away']):3d}")

    # --- 6. Verificacao de completude ---
    print(f"\n6. Verificacao de completude:")
    n_total = len(cards_df)
    n_verdict = cards_df[cards_df['verdict'] != 'Sem xG'].shape[0]
    n_no_xg = cards_df[cards_df['verdict'] == 'Sem xG'].shape[0]
    print(f"   Fichas geradas:     {n_total}")
    print(f"   Com classificacao:  {n_verdict}")
    print(f"   Sem xG:            {n_no_xg}")

    verdicts = cards_df['verdict'].value_counts()
    print(f"\n   Distribuicao de veredictos:")
    for v, c in verdicts.items():
        print(f"     {v:25s} {c:3d} ({c/n_total:.1%})")

    print(f"\nFase 7 concluida.")
