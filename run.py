"""BRForecast — Ponto de entrada do pipeline completo.

Uso:
    python run.py                          # Pipeline completo (temporada atual, 10k sims)
    python run.py --season 2025            # Temporada específica
    python run.py --n-sims 20000           # Mais simulações
    python run.py --backtest --season 2025 # Backtest em temporada encerrada
"""

import argparse
import os
import time

from src.config import (
    TARGET_YEAR, N_SIMULATIONS, N_SIMULATIONS_PROD, DATA_DIR,
    MIN_MATCHES_SEASON,
)


def run_pipeline(season_year, n_sims, backtest_mode=False, backtest_rounds=None):
    """Executa o pipeline completo: ELO → Poisson → Load Data → Monte Carlo."""
    from src.elo import load_historical_matches, calculate_all_elos
    from src.poisson import load_serie_a_with_xg, calculate_team_strengths
    from src.load_data import (
        get_season_id, load_current_table, load_remaining_matches, export_all,
    )
    from src.simulation import (
        run_monte_carlo, aggregate_results, export_results, backtest,
    )

    print("=" * 60)
    print(f"BRForecast — Pipeline {'Backtest' if backtest_mode else 'Completo'}")
    print(f"Temporada: {season_year} | Simulações: {n_sims}")
    print("=" * 60)

    t0 = time.time()

    # --- 1. ELO ---
    print(f"\n[1/5] Calculando ELOs...")
    elo_matches = load_historical_matches()
    elo_ratings, elo_history = calculate_all_elos(elo_matches)
    print(f"       {len(elo_ratings)} times com ELO")

    # --- 2. Poisson ---
    print(f"\n[2/5] Calculando forças Poisson...")
    matches_xg = load_serie_a_with_xg()
    target_m = matches_xg[matches_xg['season_year'] == season_year]
    target_done = target_m[target_m['status'] == 'complete']

    if len(target_done) < MIN_MATCHES_SEASON:
        print(f"       Apenas {len(target_done)} jogos em {season_year}, "
              f"incluindo {season_year - 1}")
        expanded = matches_xg[matches_xg['season_year'] >= season_year - 1]
        team_strengths, league_avgs = calculate_team_strengths(expanded)
    else:
        team_strengths, league_avgs = calculate_team_strengths(target_done)
    print(f"       {len(team_strengths)} times, "
          f"métrica: {league_avgs['metric']}")

    # --- 3. Dados da temporada ---
    print(f"\n[3/5] Carregando dados da temporada {season_year}...")
    season_id = get_season_id(season_year)
    table = load_current_table(season_id)
    remaining = load_remaining_matches(season_id)
    print(f"       {len(table)} times, {len(remaining)} jogos restantes")

    # Export CSVs da Fase 4
    export_all(season_id, elo_ratings, team_strengths, league_avgs)

    if backtest_mode:
        # --- Backtest ---
        if backtest_rounds is None:
            backtest_rounds = [10, 15, 20, 25, 30, 35]
        print(f"\n[4/5] Backtest (rodadas: {backtest_rounds})...")
        bt = backtest(season_year, backtest_rounds, n_simulations=n_sims)

        print(f"\n{'Rodada':>6s} {'Restantes':>9s} {'P(real)':>8s} "
              f"{'Top pick':>20s} {'OK?':>4s}")
        print(f"{'-'*55}")
        for _, r in bt.iterrows():
            print(f"{r['start_round']:6d} {r['remaining_games']:9d} "
                  f"{r['p_real_champion_title']:>7.1%} "
                  f"{r['predicted_champion']:>20s} "
                  f"{'SIM' if r['champion_correct'] else 'NAO':>4s}")

        elapsed = time.time() - t0
        print(f"\nBacktest concluído em {elapsed:.1f}s")
        return bt

    # --- 4. Monte Carlo ---
    print(f"\n[4/5] Rodando {n_sims} simulações Monte Carlo...")
    positions, points, team_names = run_monte_carlo(
        n_sims, remaining, table, team_strengths, league_avgs, elo_ratings,
    )

    # --- 5. Agregação e export ---
    print(f"\n[5/5] Agregando e exportando resultados...")
    results = aggregate_results(positions, points, team_names, n_sims)
    path = export_results(results, year=season_year)

    # Exibir resultado
    print(f"\n{'':3s}{'Time':25s} {'Titulo':>7s} {'Liberta':>8s} "
          f"{'Sulam':>7s} {'Rebx':>7s} {'Pts':>5s} {'Pos':>4s}")
    print(f"   {'-'*68}")
    for _, r in results.iterrows():
        print(f"   {r['team']:25s} {r['p_titulo']:>6.1%} "
              f"{r['p_libertadores']:>7.1%} {r['p_sulamericana']:>6.1%} "
              f"{r['p_rebaixamento']:>6.1%} {r['pts_mean']:>5.1f} "
              f"{r['pos_mean']:>4.1f}")

    elapsed = time.time() - t0
    print(f"\nPipeline concluído em {elapsed:.1f}s")
    print(f"Resultados exportados: {path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="BRForecast — Pipeline de simulação")
    parser.add_argument("--season", type=int, default=TARGET_YEAR,
                        help=f"Ano da temporada (default: {TARGET_YEAR})")
    parser.add_argument("--n-sims", type=int, default=N_SIMULATIONS,
                        help=f"Número de simulações (default: {N_SIMULATIONS})")
    parser.add_argument("--backtest", action="store_true",
                        help="Modo backtest (simula temporada encerrada)")
    parser.add_argument("--rounds", type=int, nargs="*",
                        help="Rodadas iniciais para backtest (ex: 10 15 20 25 30)")
    args = parser.parse_args()

    run_pipeline(
        season_year=args.season,
        n_sims=args.n_sims,
        backtest_mode=args.backtest,
        backtest_rounds=args.rounds,
    )


if __name__ == "__main__":
    main()
