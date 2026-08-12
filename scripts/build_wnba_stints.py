"""
Aggregate WNBA possession-level data into STINT-TRIPS.

A "stint" = continuous stretch with the same 10 players on the floor.
A "stint-trip" = same 10 players + same offensive team — i.e. all possessions
within a stint where one specific team had the ball, aggregated.

Within a stint, possessions alternate offense.  Each of those alternations is
a separate observation of the same matchup, so we aggregate each team's
possessions independently:

  Stint X (10 players):  6 possessions total
    → row A: team A offense, sum of 3 possessions where A had the ball
    → row B: team B offense, sum of 3 possessions where B had the ball

These two rows are independent regression observations and dramatically
reduce noise vs. one-possession-per-row.

Output: data/wnba_stints_all.csv
  Same columns as wnba_possessions_all.csv, but:
    - points, possessions, fga, fga3, ftm, fta, off_reb, def_reb, turnovers
      are SUMMED across the trip
    - lineup_complete = 1 only if every constituent possession was complete
    - new column: stint_id  (same 10 players → same stint_id)
    - new column: n_poss    (count of possessions aggregated)

Usage:
  python scripts/build_wnba_stints.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT  = Path("data/wnba_possessions_all.csv")
DEFAULT_OUTPUT = Path("data/wnba_stints_all.csv")

OFF_COLS = [f"offensePlayer{i}Id" for i in range(1, 6)]
DEF_COLS = [f"defensePlayer{i}Id" for i in range(1, 6)]

SUM_COLS = ["points", "possessions", "fga", "fga3",
            "ftm", "fta", "off_reb", "def_reb", "turnovers"]


def _lineup_key(row: np.ndarray) -> frozenset:
    """
    Set of all 10 player IDs on the floor — order/team-independent.
    Stint changes when this set changes.
    """
    pids = [int(p) for p in row if p > 0]
    return frozenset(pids)


def aggregate_stints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stints = same 10 players on floor.  Within each stint we aggregate each
    team's offensive possessions separately, producing two regression rows
    (one per direction of play) per stint.
    """
    print(f"Aggregating {len(df):,} possessions into stint-trips …")

    df = df.reset_index(drop=True).copy()

    # Lineup signature: set of 10 player IDs (order- and team-independent)
    lineup_arr = df[OFF_COLS + DEF_COLS].values
    sigs       = [_lineup_key(r) for r in lineup_arr]

    # Stint boundary: game/period change OR lineup composition change
    game_arr   = df["gameId"].values
    period_arr = df["period"].values
    new_stint  = np.zeros(len(df), dtype=bool)
    new_stint[0] = True
    for i in range(1, len(df)):
        if (game_arr[i] != game_arr[i-1]
                or period_arr[i] != period_arr[i-1]
                or sigs[i] != sigs[i-1]):
            new_stint[i] = True
    df["stint_id"] = new_stint.cumsum()

    # Group by (stint_id, offense_team) — each stint contributes up to 2 rows
    grp_keys = ["stint_id", "offenseTeamId"]

    keep_first = OFF_COLS + DEF_COLS + [
        "gameId", "period", "defenseTeamId",
        "season", "seasonType",
    ]
    agg_dict = {c: "first" for c in keep_first}
    for c in SUM_COLS:
        agg_dict[c] = "sum"
    agg_dict["lineup_complete"] = "min"

    grouped = df.groupby(grp_keys, sort=False).agg(agg_dict)
    grouped["n_poss"] = df.groupby(grp_keys, sort=False).size()
    grouped = grouped.reset_index()

    # Note: offense_team is part of the row but the offensive 5 columns also
    # need to reflect WHICH team is on offense.  In the source data each row
    # already has the correct offensive 5 / defensive 5 for its offense team —
    # since we're grouping by offense team, the .first() aggregation preserves
    # this correctly.

    n_unique_stints = df["stint_id"].nunique()
    print(f"  → {n_unique_stints:,} unique lineup stints")
    print(f"  → {len(grouped):,} stint-trips (offense-direction rows)")
    print(f"  Compression: {len(df):,} poss → {len(grouped):,} rows "
          f"({len(grouped)/len(df)*100:.1f}%)")
    print(f"  Mean possessions per trip: {grouped['n_poss'].mean():.2f}")
    print(f"  Max  possessions per trip: {grouped['n_poss'].max()}")
    poss_dist = grouped["n_poss"].value_counts().sort_index()
    print(f"  Possessions-per-trip distribution: {dict(poss_dist.head(10))}")

    col_order = (
        OFF_COLS + DEF_COLS + SUM_COLS +
        ["gameId", "period", "offenseTeamId", "defenseTeamId",
         "lineup_complete", "season", "seasonType",
         "stint_id", "n_poss"]
    )
    return grouped[col_order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} possessions from {args.input}")

    stints = aggregate_stints(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stints.to_csv(args.output, index=False)
    print(f"\nSaved {len(stints):,} stints → {args.output}")
    print()
    print("Sample (first 10 stints):")
    print(stints.head(10)[
        ["gameId", "period", "stint_id", "n_poss",
         "offenseTeamId", "points", "possessions",
         "fga", "off_reb", "def_reb", "turnovers", "lineup_complete"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
