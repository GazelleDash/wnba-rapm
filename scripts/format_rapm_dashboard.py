"""
Reshape v4 RAPM results into the nbashotcharts-style dashboard layout.

Reads:  data/wnba_rapm_results_v4.csv
Writes: data/wnba_rapm_dashboard.csv

Output columns (in order):
  name, end_year, rapm_length, total_poss,
  off_ts_val, off_tov_val, off_reb_val,
  def_ts_val, def_tov_val, def_reb_val,
  o_poss_val, d_poss_val, poss_val,
  orapm, drapm, rapm,
  rank columns for each value field above

Derived values:
  o_poss_val  = off_tov_val + off_reb_val   (possession-control only: TOV% + Reb%)
  d_poss_val  = def_tov_val + def_reb_val   (possession-control only: TOV% + Reb%)
  poss_val    = o_poss_val + d_poss_val     (net possession-margin value)
  rapm_length = numeric window size (e.g. "5Y" → 5)
  *_rank      = descending rank within each (end_year, rapm_length) cohort

Note: TS% (scoring efficiency) is intentionally excluded from poss_val — it
measures what a player does WITH a possession, not whether they gain/lose one.
poss_val isolates TOV% + Reb%, the two factors that directly control who ends
up with the ball. off_ts_val/def_ts_val remain in the output as their own
columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT  = Path("data/wnba_rapm_results_v4.csv")
DEFAULT_OUTPUT = Path("data/wnba_rapm_dashboard.csv")


def reshape(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    out["name"]         = df["name"]
    out["team"]         = df["team"] if "team" in df.columns else ""
    out["end_year"]     = df["end_year"].astype(int)
    out["rapm_length"]  = df["window"].str.replace("Y", "").astype(int)
    out["total_poss"]   = df["total_poss"].round(0).astype(int)

    out["off_ts_val"]   = df["off_ts_pts"]
    out["off_tov_val"]  = df["off_tov_pts"]
    out["off_reb_val"]  = df["off_reb_pts"]

    out["def_ts_val"]   = df["def_ts_pts"]
    out["def_tov_val"]  = df["def_tov_pts"]
    out["def_reb_val"]  = df["def_reb_pts"]

    out["o_poss_val"]   = (out["off_tov_val"] + out["off_reb_val"]).round(2)
    out["d_poss_val"]   = (out["def_tov_val"] + out["def_reb_val"]).round(2)
    out["poss_val"]     = (out["o_poss_val"] + out["d_poss_val"]).round(2)

    out["orapm"]        = df["ORAPM"]
    out["drapm"]        = df["DRAPM"]
    out["rapm"]         = df["RAPM"]

    # Ranks within each (end_year, rapm_length) cohort, descending.
    rank_map = {
        "off_ts_val": "off_ts_rank",
        "off_tov_val": "off_tov_rank",
        "off_reb_val": "off_reb_rank",
        "def_ts_val": "def_ts_rank",
        "def_tov_val": "def_tov_rank",
        "def_reb_val": "def_reb_rank",
        "o_poss_val": "o_poss_rank",
        "d_poss_val": "d_poss_rank",
        "poss_val": "poss_rank",
        "orapm": "orapm_rank",
        "drapm": "drapm_rank",
        "rapm": "rapm_rank",
    }
    grp = out.groupby(["end_year", "rapm_length"])
    for value_col, rank_col in rank_map.items():
        out[rank_col] = grp[value_col].rank(ascending=False, method="min").astype(int)

    ordered_cols = [
        "name", "team", "end_year", "rapm_length", "total_poss",
        "orapm", "rapm", "drapm",
        "off_ts_val", "off_tov_val", "off_reb_val",
        "def_ts_val", "def_tov_val", "def_reb_val",
        "o_poss_val", "d_poss_val", "poss_val",
        "off_ts_rank", "off_tov_rank", "off_reb_rank",
        "def_ts_rank", "def_tov_rank", "def_reb_rank",
        "o_poss_rank", "d_poss_rank", "poss_rank",
        "orapm_rank", "drapm_rank", "rapm_rank",
    ]
    out = out[ordered_cols]

    # Sort: latest end_year first, smallest window first, then by rapm rank
    out = out.sort_values(
        ["end_year", "rapm_length", "rapm_rank"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")

    out = reshape(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Wrote {len(out):,} rows → {args.output}\n")
    print("Columns:", list(out.columns))
    print("\nTop 10 (end_year=max, 5Y window):")
    yr = out["end_year"].max()
    preview = out[(out["end_year"] == yr) & (out["rapm_length"] == 5)].head(10)
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
