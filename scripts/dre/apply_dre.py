"""
Apply the fitted WNBA DRE formula to box scores — NO RAPM required.

This is the point of DRE: the RAPM regression happens ONCE (build_wnba_dre.py)
to learn the weights; after that the metric is a pure box-score linear formula.
So this script can score:
  • any player, including ones with too few minutes to have a RAPM estimate
  • any single game (the original "DAILY RAPM Estimate" use case)
  • a brand-new season the moment box scores post, with no RAPM refit

Two granularities:
  --level game     one row per player-game   (daily DRE, Ferrigan's intent)
  --level season   one row per player-season (default)

Possession handling mirrors the fitting script exactly:
  team_poss_game   = FGA - ORB + TOV + 0.44*FTA          (team totals, that game)
  game_duration    = team_minutes / 5                    (5 players always on floor)
  player_poss      = team_poss * player_min / game_duration

Outputs both scales:
  dre             — Ferrigan convention (PTS=1), Game-Score-style per-100 rate
  dre_rapm_scale  — raw regression scale, directly comparable to our RAPM
  dre_total       — bulk production: dre * poss / 100 (season) or per game

Usage:
  python scripts/dre/apply_dre.py                          # 2026 season-level
  python scripts/dre/apply_dre.py --season 2026 --level game
  python scripts/dre/apply_dre.py --seasons 2024 2025 2026
  python scripts/dre/apply_dre.py --min-poss 300           # filter output
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_wnba_dre import WEHOOP_BOX_URL, BOX_STAT_COLS, RATE_LABELS

COEF_JSON = Path("data/dre/wnba_dre_coefficients.json")
OUT_DIR   = Path("data/dre")


def load_coefs() -> dict:
    if not COEF_JSON.exists():
        raise SystemExit(
            f"No coefficients at {COEF_JSON}.\n"
            f"Run:  python scripts/dre/build_wnba_dre.py   (fits weights once)"
        )
    with open(COEF_JSON) as f:
        return json.load(f)


def load_box_raw(season: int) -> pd.DataFrame | None:
    """Regular-season box rows that actually played, with possessions attached."""
    try:
        r = requests.get(WEHOOP_BOX_URL.format(season=season), timeout=60)
        if r.status_code != 200:
            return None
        df = pd.read_parquet(io.BytesIO(r.content))
    except Exception as e:
        print(f"  {season}: fetch failed ({e})")
        return None

    df = df[(df["season_type"] == 2) & (df["did_not_play"] == False)].copy()  # noqa: E712
    df = df[df["minutes"].fillna(0) > 0]
    if df.empty:
        return None

    df["fg2a"] = df["field_goals_attempted"] - df["three_point_field_goals_attempted"]

    team_tot = df.groupby(["game_id", "team_id"]).agg(
        team_min=("minutes", "sum"),
        team_fga=("field_goals_attempted", "sum"),
        team_fta=("free_throws_attempted", "sum"),
        team_orb=("offensive_rebounds", "sum"),
        team_tov=("turnovers", "sum"),
    ).reset_index()
    team_tot["team_poss"] = (team_tot["team_fga"] - team_tot["team_orb"]
                             + team_tot["team_tov"] + 0.44 * team_tot["team_fta"])
    df = df.merge(team_tot[["game_id", "team_id", "team_min", "team_poss"]],
                  on=["game_id", "team_id"], how="left")
    game_duration = df["team_min"] / 5.0
    df["player_poss"] = np.where(
        game_duration > 0, df["team_poss"] * df["minutes"] / game_duration, 0.0)
    df["season"] = season
    return df


def score(df: pd.DataFrame, coefs: dict) -> pd.DataFrame:
    """Attach per-100 rates + DRE columns. df must have raw counts and 'poss'."""
    order = coefs["feature_order"]
    scaled = np.array([coefs["scaled"][k] for k in order])
    raw    = np.array([coefs["raw"][k] for k in order])

    rate_cols = []
    for c in BOX_STAT_COLS:
        lbl = RATE_LABELS[c].lower()
        col = f"{lbl}100"
        df[col] = np.where(df["poss"] > 0, df[lbl] / df["poss"] * 100.0, 0.0)
        rate_cols.append(col)

    X = df[rate_cols].values
    df["dre"] = (X @ scaled + coefs["scaled_intercept"]).round(3)
    df["dre_rapm_scale"] = (X @ raw + coefs["raw_intercept"]).round(3)
    df["dre_total"] = (df["dre"] * df["poss"] / 100.0).round(2)
    return df


def main() -> None:
    today = datetime.date.today()
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--seasons", type=int, nargs="+", default=None)
    ap.add_argument("--level", choices=["season", "game"], default="season")
    ap.add_argument("--min-poss", type=float, default=0.0, dest="min_poss",
                    help="Filter OUTPUT rows below this possession count")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    seasons = args.seasons or [args.season or today.year]
    coefs = load_coefs()
    print(f"DRE weights: fit on {coefs['fit_target']}, "
          f"R²={coefs['r2_insample']}, n={coefs['n_train']:,}")
    print(f"Scoring {seasons} at {args.level} level (box scores only — no RAPM)\n")

    frames = []
    for s in seasons:
        raw = load_box_raw(s)
        if raw is None or raw.empty:
            print(f"  {s}: no data")
            continue

        if args.level == "game":
            d = raw.rename(columns={
                "athlete_display_name": "name",
                "team_abbreviation": "team",
                "player_poss": "poss",
                **{c: RATE_LABELS[c].lower() for c in BOX_STAT_COLS},
            })
            keep = ["name", "team", "season", "game_id", "game_date", "minutes", "poss"]
            d = d[keep + [RATE_LABELS[c].lower() for c in BOX_STAT_COLS]]
        else:
            d = raw.groupby(["athlete_display_name", "season"]).agg(
                team=("team_abbreviation", lambda x: x.mode().iloc[0] if len(x) else ""),
                games=("game_id", "nunique"),
                minutes=("minutes", "sum"),
                poss=("player_poss", "sum"),
                **{RATE_LABELS[c].lower(): (c, "sum") for c in BOX_STAT_COLS},
            ).reset_index().rename(columns={"athlete_display_name": "name"})

        d = score(d, coefs)
        frames.append(d)
        print(f"  {s}: {len(d):,} rows scored")

    if not frames:
        raise SystemExit("No data scored.")
    out = pd.concat(frames, ignore_index=True)

    if args.min_poss:
        before = len(out)
        out = out[out["poss"] >= args.min_poss]
        print(f"\n  Filtered poss ≥ {args.min_poss:.0f}: {before:,} → {len(out):,}")

    sort_keys = ["season", "dre"] if args.level == "season" else ["game_date", "dre"]
    out = out.sort_values(sort_keys, ascending=[False, False]).reset_index(drop=True)
    if args.level == "season":
        out["dre_rank"] = out.groupby("season")["dre"].rank(
            ascending=False, method="min").astype(int)

    path = args.output or (OUT_DIR / f"wnba_dre_{args.level}"
                           f"{'_' + str(seasons[0]) if len(seasons) == 1 else ''}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\nSaved {len(out):,} rows → {path}")

    show = ["name", "team", "poss", "dre", "dre_rapm_scale", "dre_total"]
    if args.level == "game":
        show = ["name", "team", "game_date", "minutes", "dre", "dre_total"]
    print(f"\n=== Top 15 ({args.level} level) ===")
    print(out.head(15)[show].to_string(index=False))


if __name__ == "__main__":
    main()
