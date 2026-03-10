"""Bracket rendering and visualization.

Console-only output: advancement tables, championship probabilities,
Final Four matchups, and region-by-region bracket trees.
"""

import logging

from src.bracket.predictor import BracketPrediction

logger = logging.getLogger(__name__)

# Round labels
_ROUND_NAMES = {
    0: "Play-in",
    1: "R64",
    2: "R32",
    3: "S16",
    4: "E8",
    5: "F4",
    6: "Champ",
}

# Region labels for display
_REGION_NAMES = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}


def print_bracket(prediction: BracketPrediction) -> None:
    """Print region-by-region bracket with EV picks and confidence.

    Shows each round's picks organized by region.
    """
    bracket = prediction.bracket
    picks_by_slot = {p["slot"]: p for p in prediction.picks}
    regions = sorted(set(
        slot_name[2] if slot_name.startswith("R") and len(slot_name) > 2 else ""
        for slot_name in bracket.slots
    ) - {""})

    print(f"\n{'='*70}")
    print(f"  {prediction.season} NCAA Tournament Bracket Prediction")
    print(f"  ({prediction.simulation.n_simulations:,} MC simulations)")
    print(f"{'='*70}")

    for region in regions:
        region_name = _REGION_NAMES.get(region, region)
        print(f"\n  --- {region_name} Region ---")

        for rnd in range(1, 5):
            round_name = _ROUND_NAMES.get(rnd, f"R{rnd}")
            region_slots = [
                s for s in bracket.round_slots.get(rnd, [])
                if len(s) > 2 and s[2] == region
            ]
            if not region_slots:
                continue

            print(f"    {round_name}:")
            for slot_name in sorted(region_slots):
                pick = picks_by_slot.get(slot_name)
                if pick:
                    print(
                        f"      {slot_name}: ({pick['seed']:>2}) {pick['team_name']:<20s} "
                        f"[{pick['confidence']*100:5.1f}%]"
                    )

    # Final Four and Championship
    print(f"\n  --- Final Four ---")
    for rnd in [5, 6]:
        round_name = _ROUND_NAMES.get(rnd, f"R{rnd}")
        slots = bracket.round_slots.get(rnd, [])
        print(f"    {round_name}:")
        for slot_name in sorted(slots):
            pick = picks_by_slot.get(slot_name)
            if pick:
                print(
                    f"      {slot_name}: ({pick['seed']:>2}) {pick['team_name']:<20s} "
                    f"[{pick['confidence']*100:5.1f}%]"
                )

    print()


def print_advancement_table(
    prediction: BracketPrediction,
    min_round: int = 1,
) -> None:
    """Print advancement probability table for all teams.

    Columns: Team | Seed | R64 | R32 | S16 | E8 | F4 | Champ

    Args:
        prediction: BracketPrediction with simulation results.
        min_round: Minimum round to start showing (1=R64).
    """
    bracket = prediction.bracket
    sim = prediction.simulation
    rounds_to_show = [r for r in range(min_round, 7)]

    # Build team info
    team_info: dict[int, tuple[str, int]] = {}  # team_id → (name, seed)
    for seed_str, (tid, tname, snum) in bracket.teams.items():
        if tid not in team_info or snum < team_info[tid][1]:
            team_info[tid] = (tname, snum)

    # Build rows with advancement probs
    rows = []
    for team_id, adv in sim.advancement_probs.items():
        name, seed = team_info.get(team_id, (f"Team{team_id}", 0))
        row = {"team_id": team_id, "team": name, "seed": seed}
        for rnd in rounds_to_show:
            row[_ROUND_NAMES.get(rnd, f"R{rnd}")] = adv.get(rnd, 0.0)
        rows.append(row)

    # Sort by championship probability descending, then seed
    rows.sort(key=lambda r: (-r.get("Champ", 0), r["seed"], r["team"]))

    # Print header
    header_parts = [f"{'Team':<24s} {'Seed':>4s}"]
    for rnd in rounds_to_show:
        header_parts.append(f"{_ROUND_NAMES.get(rnd, f'R{rnd}'):>6s}")
    header = " | ".join(header_parts)

    print(f"\n{'='*len(header)}")
    print(f"  Advancement Probabilities — {prediction.season}")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for row in rows:
        parts = [f"{row['team']:<24s} {row['seed']:>4d}"]
        for rnd in rounds_to_show:
            col = _ROUND_NAMES.get(rnd, f"R{rnd}")
            prob = row.get(col, 0.0)
            if prob >= 0.995:
                parts.append(f"{'99+%':>6s}")
            elif prob < 0.005:
                parts.append(f"{'  — ':>6s}")
            else:
                parts.append(f"{prob*100:5.1f}%")
        print(" | ".join(parts))

    print()


def print_champion_probabilities(
    prediction: BracketPrediction,
    top_n: int = 16,
) -> None:
    """Print top N teams by championship probability.

    Args:
        prediction: BracketPrediction with simulation results.
        top_n: Number of teams to show.
    """
    bracket = prediction.bracket
    sim = prediction.simulation

    # Build team info
    team_info: dict[int, tuple[str, int]] = {}
    for seed_str, (tid, tname, snum) in bracket.teams.items():
        if tid not in team_info or snum < team_info[tid][1]:
            team_info[tid] = (tname, snum)

    # Sort by championship probability
    sorted_teams = sorted(
        sim.championship_probs.items(),
        key=lambda x: -x[1],
    )[:top_n]

    print(f"\n{'='*50}")
    print(f"  Championship Probabilities — {prediction.season}")
    print(f"{'='*50}")
    print(f"  {'Rank':>4s}  {'Team':<24s} {'Seed':>4s}  {'Prob':>7s}")
    print(f"  {'-'*44}")

    for rank, (team_id, prob) in enumerate(sorted_teams, 1):
        name, seed = team_info.get(team_id, (f"Team{team_id}", 0))
        bar = "#" * int(prob * 100)
        print(f"  {rank:>4d}. ({seed:>2d}) {name:<20s}  {prob*100:6.2f}%  {bar}")

    # Show total
    total = sum(sim.championship_probs.values())
    print(f"\n  Championship probs sum: {total:.4f}")
    print()


def print_final_four(prediction: BracketPrediction) -> None:
    """Print Final Four matchups with probabilities."""
    print(f"\n{'='*50}")
    print(f"  Final Four — {prediction.season}")
    print(f"{'='*50}")

    if not prediction.final_four:
        print("  No Final Four data available.")
        print()
        return

    for team in prediction.final_four:
        slot = team.get("slot", "?")
        print(
            f"  {slot}: ({team['seed']:>2d}) {team['team_name']:<20s} "
            f"[{team['probability']*100:5.1f}% to win semifinal]"
        )

    # Champion
    if prediction.champion:
        champ = prediction.champion
        print(f"\n  Champion: ({champ['seed']:>2d}) {champ['team_name']}")
        print(f"  Championship probability: {champ['probability']*100:.2f}%")

    print()


def print_round1_decomposition(prediction: BracketPrediction) -> None:
    """Print expert decomposition for Round 1 games.

    Shows each expert's prediction and gating weight for every R1 matchup.
    """
    df = prediction.round1_decomposed
    if df is None or df.empty:
        print("  No R1 decomposition available.")
        return

    print(f"\n{'='*90}")
    print(f"  Round 1 Expert Decomposition — {prediction.season}")
    print(f"{'='*90}")
    print(
        f"  {'Matchup':<35s} "
        f"{'P_seed':>6s} {'P_eff':>6s} {'P_unc':>6s}  "
        f"{'W_seed':>6s} {'W_eff':>6s} {'W_unc':>6s}  "
        f"{'Blend':>6s}"
    )
    print(f"  {'-'*86}")

    for _, row in df.iterrows():
        matchup = f"({row['seed_a']:>2d}) {row['team_a']:<12s} v ({row['seed_b']:>2d}) {row['team_b']:<12s}"
        print(
            f"  {matchup:<35s} "
            f"{row['p_seed']:6.3f} {row['p_eff']:6.3f} {row['p_unc']:6.3f}  "
            f"{row['w_seed']:6.3f} {row['w_eff']:6.3f} {row['w_unc']:6.3f}  "
            f"{row['p_blend']:6.3f}"
        )

    print()
