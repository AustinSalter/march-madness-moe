"""Full bracket generation from MOE + MC simulation.

Orchestrates: bracket loading → TeamStatsLookup → MC simulation → pick extraction.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.bracket.structure import Bracket, load_bracket
from src.config import MC_SIMULATIONS, MC_RANDOM_SEED, TOURNAMENT_DIR
from src.data.kaggle_loader import load_kenpom
from src.features.context_features import build_upset_rate_lookup
from src.features.kenpom_deltas import compute_deltas
from src.data.merge import merge_kenpom_with_matchups
from src.simulation.matchup_builder import TeamStatsLookup
from src.simulation.mc_engine import MCSimulator, SimulationResult

logger = logging.getLogger(__name__)


@dataclass
class BracketPrediction:
    """Complete bracket prediction with metadata."""
    season: int
    bracket: Bracket
    simulation: SimulationResult
    picks: list[dict] = field(default_factory=list)
    champion: dict = field(default_factory=dict)
    final_four: list[dict] = field(default_factory=list)
    round1_decomposed: pd.DataFrame | None = None


class BracketPredictor:
    """Orchestrator: MOE + MC → bracket prediction."""

    def __init__(
        self,
        moe,
        n_simulations: int = MC_SIMULATIONS,
        random_seed: int = MC_RANDOM_SEED,
    ):
        """Initialize predictor with a trained MOE.

        Args:
            moe: Trained MOEEnsemble instance.
            n_simulations: Number of MC simulations.
            random_seed: Random seed for reproducibility.
        """
        self.moe = moe
        self.n_simulations = n_simulations
        self.random_seed = random_seed

    def predict(
        self,
        season: int,
        kenpom_df: pd.DataFrame | None = None,
        upset_rate_lookup: dict | None = None,
        tournament_dir: Path = TOURNAMENT_DIR,
    ) -> BracketPrediction:
        """Generate bracket prediction for a season.

        Args:
            season: Tournament year.
            kenpom_df: KenPom data for this season. If None, loaded automatically.
            upset_rate_lookup: Historical upset rates. If None, built from
                all available historical data.
            tournament_dir: Path to tournament data directory.

        Returns:
            BracketPrediction with picks, champion, final four, etc.
        """
        # 1. Load bracket structure
        bracket = load_bracket(season, tournament_dir)

        # 2. Load KenPom if needed
        if kenpom_df is None:
            kenpom_df = load_kenpom()

        # 3. Build upset rate lookup if needed
        if upset_rate_lookup is None:
            upset_rate_lookup = self._build_default_upset_lookup()

        # 4. Build TeamStatsLookup
        lookup = TeamStatsLookup(
            season=season,
            kenpom_df=kenpom_df,
            upset_rate_lookup=upset_rate_lookup,
            tournament_dir=tournament_dir,
        )

        # 5. Build win_prob_fn closure
        def win_prob_fn(team_id_a, team_id_b, round_num):
            return lookup.get_win_prob(team_id_a, team_id_b, round_num, self.moe)

        # 6. Run MC simulation
        simulator = MCSimulator(
            n_simulations=self.n_simulations,
            random_seed=self.random_seed,
        )
        sim_result = simulator.simulate(bracket, win_prob_fn)

        # 7. Extract picks from EV bracket
        picks = self._extract_picks(bracket, sim_result)
        champion = self._extract_champion(bracket, sim_result)
        final_four = self._extract_final_four(bracket, sim_result)

        # 8. Get R1 decomposition for analysis
        round1_decomposed = self._decompose_round1(bracket, lookup)

        prediction = BracketPrediction(
            season=season,
            bracket=bracket,
            simulation=sim_result,
            picks=picks,
            champion=champion,
            final_four=final_four,
            round1_decomposed=round1_decomposed,
        )

        logger.info(
            "Bracket prediction complete: season=%d, champion=%s (%.1f%%)",
            season,
            champion.get("team_name", "?"),
            champion.get("probability", 0) * 100,
        )
        return prediction

    def predict_historical(
        self,
        season: int,
        merged_df: pd.DataFrame | None = None,
        tournament_dir: Path = TOURNAMENT_DIR,
    ) -> BracketPrediction:
        """Generate bracket prediction using historical KenPom data.

        Same flow as predict() but loads KenPom from Kaggle data.

        Args:
            season: Historical tournament year.
            merged_df: Optional pre-merged DataFrame for building upset lookup.
            tournament_dir: Path to tournament data directory.

        Returns:
            BracketPrediction.
        """
        kenpom_df = load_kenpom()

        # Build upset rate from all data except test season
        if merged_df is None:
            merged_df = merge_kenpom_with_matchups(kenpom_df=kenpom_df)

        train_df = merged_df[merged_df["season"] != season].copy()
        train_deltas = compute_deltas(train_df)
        upset_rate_lookup = build_upset_rate_lookup(train_deltas)

        return self.predict(
            season=season,
            kenpom_df=kenpom_df,
            upset_rate_lookup=upset_rate_lookup,
            tournament_dir=tournament_dir,
        )

    def _build_default_upset_lookup(self) -> dict:
        """Build upset rate lookup from all available historical data."""
        try:
            merged_df = merge_kenpom_with_matchups()
            merged_deltas = compute_deltas(merged_df)
            return build_upset_rate_lookup(merged_deltas)
        except Exception as e:
            logger.warning("Could not build upset lookup: %s, using empty", e)
            return {}

    def _extract_picks(
        self, bracket: Bracket, sim: SimulationResult,
    ) -> list[dict]:
        """Extract game-by-game picks from EV bracket."""
        picks = []
        for slot_name, team_id in sim.ev_bracket:
            slot = bracket.slots[slot_name]
            # Find team info
            seed_info = None
            for seed_str, (tid, tname, snum) in bracket.teams.items():
                if tid == team_id:
                    seed_info = (tname, snum, seed_str)
                    break

            # Get confidence from slot results
            slot_wins = sim.slot_results.get(slot_name, {})
            total = sum(slot_wins.values())
            confidence = slot_wins.get(team_id, 0) / total if total > 0 else 0

            picks.append({
                "slot": slot_name,
                "round": slot.round_num,
                "team_id": team_id,
                "team_name": seed_info[0] if seed_info else f"Team{team_id}",
                "seed": seed_info[1] if seed_info else 0,
                "confidence": confidence,
            })

        return picks

    def _extract_champion(
        self, bracket: Bracket, sim: SimulationResult,
    ) -> dict:
        """Extract predicted champion with probability."""
        if not sim.championship_probs:
            return {}

        champ_id = max(sim.championship_probs, key=sim.championship_probs.get)
        champ_prob = sim.championship_probs[champ_id]

        # Find team info
        for seed_str, (tid, tname, snum) in bracket.teams.items():
            if tid == champ_id:
                return {
                    "team_id": champ_id,
                    "team_name": tname,
                    "seed": snum,
                    "probability": champ_prob,
                }

        return {
            "team_id": champ_id,
            "team_name": f"Team{champ_id}",
            "seed": 0,
            "probability": champ_prob,
        }

    def _extract_final_four(
        self, bracket: Bracket, sim: SimulationResult,
    ) -> list[dict]:
        """Extract Final Four teams with advancement probabilities."""
        # Find R5 (Final Four) slots
        r5_slots = bracket.round_slots.get(5, [])

        final_four = []
        for slot_name in r5_slots:
            slot_wins = sim.slot_results.get(slot_name, {})
            if not slot_wins:
                continue
            # Most likely winner of each F4 game
            best_id = max(slot_wins, key=slot_wins.get)
            total = sum(slot_wins.values())
            prob = slot_wins.get(best_id, 0) / total if total > 0 else 0

            for seed_str, (tid, tname, snum) in bracket.teams.items():
                if tid == best_id:
                    final_four.append({
                        "team_id": best_id,
                        "team_name": tname,
                        "seed": snum,
                        "probability": prob,
                        "slot": slot_name,
                    })
                    break
            else:
                final_four.append({
                    "team_id": best_id,
                    "team_name": f"Team{best_id}",
                    "seed": 0,
                    "probability": prob,
                    "slot": slot_name,
                })

        return final_four

    def _decompose_round1(
        self, bracket: Bracket, lookup: TeamStatsLookup,
    ) -> pd.DataFrame | None:
        """Get MOE expert decomposition for all Round 1 games."""
        r1_slots = bracket.round_slots.get(1, [])
        if not r1_slots:
            return None

        rows = []
        for slot_name in r1_slots:
            slot = bracket.slots[slot_name]

            # Resolve seed references to team IDs
            team_a_info = bracket.get_team(slot.strong_source)
            team_b_info = bracket.get_team(slot.weak_source)
            if not team_a_info or not team_b_info:
                continue

            team_id_a, name_a, seed_a = team_a_info
            team_id_b, name_b, seed_b = team_b_info

            try:
                fs = lookup.get_matchup_features(team_id_a, team_id_b, 1)
                # Fill NaN in gating features
                for col in fs.gating_features:
                    if col in fs.X.columns:
                        fs.X[col] = fs.X[col].fillna(0)

                decomp = self.moe.predict_decomposed(fs)
                row = decomp.iloc[0].to_dict()
                row["slot"] = slot_name
                row["team_a"] = name_a
                row["seed_a"] = seed_a
                row["team_b"] = name_b
                row["seed_b"] = seed_b
                rows.append(row)
            except Exception as e:
                logger.warning("Could not decompose %s: %s", slot_name, e)

        return pd.DataFrame(rows) if rows else None
