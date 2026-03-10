"""CLI: Generate bracket prediction using trained MOE + Monte Carlo simulation.

Usage:
    python -m scripts.predict_bracket --season 2025
    python -m scripts.predict_bracket --season 2025 --log5-baseline
    python -m scripts.predict_bracket --season 2026 --n-sims 50000
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bracket.predictor import BracketPredictor
from src.bracket.visualizer import (
    print_advancement_table,
    print_bracket,
    print_champion_probabilities,
    print_final_four,
    print_round1_decomposition,
)
from src.config import MC_RANDOM_SEED, MC_SIMULATIONS, MODELS_DIR
from src.models.moe_ensemble import MOEEnsemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate NCAA tournament bracket prediction",
    )
    parser.add_argument(
        "--season", type=int, default=2025,
        help="Tournament year (default: 2025)",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help=f"Path to saved MOE model (default: {MODELS_DIR}/moe)",
    )
    parser.add_argument(
        "--n-sims", type=int, default=MC_SIMULATIONS,
        help=f"Number of MC simulations (default: {MC_SIMULATIONS})",
    )
    parser.add_argument(
        "--seed", type=int, default=MC_RANDOM_SEED,
        help=f"Random seed (default: {MC_RANDOM_SEED})",
    )
    parser.add_argument(
        "--log5-baseline", action="store_true",
        help="Also run Log5 baseline for comparison",
    )
    parser.add_argument(
        "--historical", action="store_true",
        help="Use historical KenPom data (for backtesting past seasons)",
    )
    parser.add_argument(
        "--top-n", type=int, default=16,
        help="Number of teams to show in championship probabilities (default: 16)",
    )
    parser.add_argument(
        "--no-decomposition", action="store_true",
        help="Skip R1 expert decomposition output",
    )
    args = parser.parse_args()

    # Load trained MOE
    model_dir = Path(args.model_dir) if args.model_dir else MODELS_DIR / "moe"
    logger.info("Loading MOE from %s", model_dir)
    moe = MOEEnsemble.load(model_dir)

    # Build predictor
    predictor = BracketPredictor(
        moe=moe,
        n_simulations=args.n_sims,
        random_seed=args.seed,
    )

    # Generate prediction
    if args.season >= 2026 and not args.historical:
        # Future season — try to use scraped KenPom data
        logger.info("Generating prediction for %d (live data)", args.season)
        try:
            from src.data.scraper import scrape_2026
            kenpom_df = scrape_2026()
            prediction = predictor.predict(season=args.season, kenpom_df=kenpom_df)
        except Exception as e:
            logger.error("Could not load 2026 data: %s", e)
            logger.info("Falling back to historical data if available")
            prediction = predictor.predict_historical(season=args.season)
    elif args.historical or args.season <= 2025:
        # Historical season
        logger.info("Generating prediction for %d (historical data)", args.season)
        prediction = predictor.predict_historical(season=args.season)
    else:
        prediction = predictor.predict(season=args.season)

    # Print results
    print_champion_probabilities(prediction, top_n=args.top_n)
    print_final_four(prediction)
    print_advancement_table(prediction)
    print_bracket(prediction)

    if not args.no_decomposition:
        print_round1_decomposition(prediction)

    # Optional Log5 baseline comparison
    if args.log5_baseline:
        _run_log5_comparison(args, predictor.n_simulations, predictor.random_seed)


def _run_log5_comparison(args, n_sims, random_seed):
    """Run Log5 baseline and print comparison."""
    from src.bracket.structure import load_bracket
    from src.simulation.log5 import build_log5_win_prob_fn
    from src.simulation.mc_engine import MCSimulator

    logger.info("Running Log5 baseline for comparison...")

    bracket = load_bracket(args.season)
    log5_fn = build_log5_win_prob_fn(args.season)

    simulator = MCSimulator(n_simulations=n_sims, random_seed=random_seed)
    log5_sim = simulator.simulate(bracket, log5_fn)

    print(f"\n{'='*50}")
    print(f"  Log5 Baseline — {args.season}")
    print(f"{'='*50}")

    # Top 10 championship probabilities
    team_info = {}
    for seed_str, (tid, tname, snum) in bracket.teams.items():
        if tid not in team_info or snum < team_info[tid][1]:
            team_info[tid] = (tname, snum)

    sorted_teams = sorted(log5_sim.championship_probs.items(), key=lambda x: -x[1])[:10]
    print(f"  {'Rank':>4s}  {'Team':<24s} {'Seed':>4s}  {'Log5':>7s}")
    print(f"  {'-'*44}")
    for rank, (team_id, prob) in enumerate(sorted_teams, 1):
        name, seed = team_info.get(team_id, (f"Team{team_id}", 0))
        print(f"  {rank:>4d}. ({seed:>2d}) {name:<20s}  {prob*100:6.2f}%")
    print()


if __name__ == "__main__":
    main()
