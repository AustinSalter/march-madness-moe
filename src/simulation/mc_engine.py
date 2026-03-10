"""Monte Carlo tournament bracket simulation engine.

Runs N simulated tournaments, each time drawing Bernoulli outcomes for every
game using win probabilities from a provided function (MOE, Log5, etc.).
Aggregates advancement probabilities, expected-value bracket, and championship odds.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from src.bracket.structure import Bracket, BracketSlot, is_seed_reference, get_slot_round_order
from src.config import MC_SIMULATIONS, MC_RANDOM_SEED

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Aggregated results from Monte Carlo bracket simulation."""
    n_simulations: int
    advancement_probs: dict[int, dict[int, float]] = field(default_factory=dict)
    ev_bracket: list[tuple[str, int]] = field(default_factory=list)
    slot_results: dict[str, dict[int, int]] = field(default_factory=dict)
    championship_probs: dict[int, float] = field(default_factory=dict)


class MCSimulator:
    """Monte Carlo bracket simulator.

    Pre-generates random numbers for all simulations, caches win probabilities
    for unique matchups, and resolves the bracket graph in round order.
    """

    def __init__(
        self,
        n_simulations: int = MC_SIMULATIONS,
        random_seed: int = MC_RANDOM_SEED,
    ):
        self.n_simulations = n_simulations
        self.random_seed = random_seed

    def simulate(
        self,
        bracket: Bracket,
        win_prob_fn: callable,
    ) -> SimulationResult:
        """Run Monte Carlo simulation of the full tournament.

        Args:
            bracket: Bracket structure with slots and team assignments.
            win_prob_fn: (team_id_a, team_id_b, round_num) → P(team_a wins).

        Returns:
            SimulationResult with advancement probs, EV bracket, etc.
        """
        rng = np.random.default_rng(self.random_seed)
        n_slots = len(bracket.slots)

        # Pre-generate all random draws
        randoms = rng.random((self.n_simulations, n_slots))

        # Probability cache: (min_id, max_id, round) → P(lower_id wins)
        prob_cache: dict[tuple[int, int, int], float] = {}

        # Track per-slot winners across simulations
        slot_results: dict[str, dict[int, int]] = {
            slot_name: {} for slot_name in bracket.slots
        }

        # Track per-team advancement: team_id → {round → count}
        advancement_counts: dict[int, dict[int, int]] = {}

        # Get rounds in execution order
        round_order = get_slot_round_order(bracket)

        # Build ordered list of slots for random index mapping
        slot_order = []
        for rnd in round_order:
            if rnd in bracket.round_slots:
                slot_order.extend(bracket.round_slots[rnd])

        slot_to_idx = {name: i for i, name in enumerate(slot_order)}

        logger.info(
            "Starting MC simulation: %d sims, %d slots",
            self.n_simulations, n_slots,
        )

        for sim_idx in range(self.n_simulations):
            # Simulate one full tournament
            sim_winners = self._simulate_one(
                bracket, win_prob_fn, prob_cache,
                randoms[sim_idx], slot_to_idx,
            )

            # Record results
            for slot_name, winner_id in sim_winners.items():
                slot_round = bracket.slots[slot_name].round_num
                # Count slot winner
                if winner_id not in slot_results[slot_name]:
                    slot_results[slot_name][winner_id] = 0
                slot_results[slot_name][winner_id] += 1

                # Count advancement
                if winner_id not in advancement_counts:
                    advancement_counts[winner_id] = {}
                # Winning a round-R game means advancing past round R
                adv_round = slot_round
                if adv_round not in advancement_counts[winner_id]:
                    advancement_counts[winner_id][adv_round] = 0
                advancement_counts[winner_id][adv_round] += 1

        # Compute probabilities
        n = self.n_simulations
        advancement_probs = {}
        for team_id, rounds in advancement_counts.items():
            advancement_probs[team_id] = {
                rnd: count / n for rnd, count in rounds.items()
            }

        # Championship probabilities
        championship_probs = {}
        champ_slot = None
        for slot_name, slot in bracket.slots.items():
            if slot.round_num == 6:
                champ_slot = slot_name
                break

        if champ_slot and champ_slot in slot_results:
            for team_id, count in slot_results[champ_slot].items():
                championship_probs[team_id] = count / n

        # EV bracket: pick most-likely winner at each slot
        ev_bracket = []
        for slot_name in slot_order:
            if slot_results[slot_name]:
                best_team = max(
                    slot_results[slot_name],
                    key=slot_results[slot_name].get,
                )
                ev_bracket.append((slot_name, best_team))

        logger.info(
            "MC simulation complete: %d unique matchups cached",
            len(prob_cache),
        )

        return SimulationResult(
            n_simulations=n,
            advancement_probs=advancement_probs,
            ev_bracket=ev_bracket,
            slot_results=slot_results,
            championship_probs=championship_probs,
        )

    def _simulate_one(
        self,
        bracket: Bracket,
        win_prob_fn: callable,
        prob_cache: dict[tuple[int, int, int], float],
        randoms: np.ndarray,
        slot_to_idx: dict[str, int],
    ) -> dict[str, int]:
        """Simulate a single tournament. Returns slot_name → winner_team_id."""
        winners: dict[str, int] = {}
        round_order = get_slot_round_order(bracket)

        for rnd in round_order:
            if rnd not in bracket.round_slots:
                continue
            for slot_name in bracket.round_slots[rnd]:
                slot = bracket.slots[slot_name]

                # Resolve team IDs from sources
                team_strong = self._resolve_team(
                    slot.strong_source, bracket, winners,
                )
                team_weak = self._resolve_team(
                    slot.weak_source, bracket, winners,
                )

                if team_strong is None or team_weak is None:
                    # Can't resolve — skip (shouldn't happen in valid bracket)
                    continue

                # Get win probability (cached)
                p_strong_wins = self._get_cached_prob(
                    team_strong, team_weak, slot.round_num,
                    win_prob_fn, prob_cache,
                )

                # Draw outcome
                idx = slot_to_idx[slot_name]
                if randoms[idx] < p_strong_wins:
                    winners[slot_name] = team_strong
                else:
                    winners[slot_name] = team_weak

        return winners

    def _resolve_team(
        self,
        source: str,
        bracket: Bracket,
        winners: dict[str, int],
    ) -> int | None:
        """Resolve a source string to a team ID.

        Seed refs (e.g., 'W01') → look up in bracket.teams.
        Slot refs (e.g., 'R1W1') → look up in simulation winners.

        Play-in slots (e.g., 'W16', 'Y11') look like seed refs but are also
        slot names — check winners first, then bracket.teams.
        """
        # Always check winners first (handles play-in slots like "W16")
        if source in winners:
            return winners[source]
        if is_seed_reference(source):
            team_info = bracket.get_team(source)
            return team_info[0] if team_info else None
        return None

    def _get_cached_prob(
        self,
        team_a: int,
        team_b: int,
        round_num: int,
        win_prob_fn: callable,
        prob_cache: dict,
    ) -> float:
        """Get P(team_a wins) with caching.

        Cache key uses (min_id, max_id, round) and stores P(lower_id wins).
        """
        lo, hi = min(team_a, team_b), max(team_a, team_b)
        cache_key = (lo, hi, round_num)

        if cache_key not in prob_cache:
            # Call win_prob_fn: P(lo wins over hi)
            prob_cache[cache_key] = win_prob_fn(lo, hi, round_num)

        p_lo_wins = prob_cache[cache_key]

        if team_a == lo:
            return p_lo_wins
        else:
            return 1.0 - p_lo_wins
