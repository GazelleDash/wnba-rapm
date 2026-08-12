"""
WNBA DRE (Daily RAPM Estimate) — Kevin Ferrigan's linear-weights metric,
rebuilt for the WNBA against OUR RAPM.

Original method (Nylon Calculus, 2015 + 2017 update):
  Regress per-100-possession box-score rates against multi-year RAPM.
  The resulting weighted-regression coefficients ARE the "DRE formula" —
  a transparent, empirically-derived alternative to arbitrary linear weights
  (Game Score, etc.), grounded in what actually predicts on-court impact.

Our adaptation:
  Target  (Y): v6 windowed RAPM, 3-year window ending in each season
                (data/wnba_rapm_dashboard_v6.csv, rapm_length==3).
                Multi-year = low-noise "truth," same rationale as the
                original NBA version and Falkenheim's 2026 WNBA RAPM work
                (which also found ~3yr windows optimal for this league).
  Inputs  (X): per-100-possession box rates for the SAME single season,
                built from ESPN player_box (sportsdataverse/wehoop-wnba-data):
                  PTS, FG2A, FG3A, FTA, ORB, DRB, AST, STL, BLK, TOV, PF
                (FGA split into 2PA/3PA and TRB split into ORB/DRB, matching
                Ferrigan's 2017 update rather than the simpler 2015 version.)
  Weight     : player's estimated season possessions (mirrors Ferrigan's
                "weight by minutes played" — we weight by possessions, the
                more precise on-court-exposure measure our data affords).
  Regression : Weighted OLS (sklearn LinearRegression, sample_weight=poss).
  Scaling    : Coefficients divided through so PTS == 1.00, exactly as
                Ferrigan did.

Possession estimation (self-contained within player_box, no join needed to
our own possession data):
  team_poss_game  = FGA - ORB + TOV + 0.44*FTA        (team box totals, that game)
  player_poss_game = team_poss_game * (player_min / team_min_that_game)
  player_poss_season = Σ player_poss_game

Output: data/dre/wnba_dre.csv
  One row per player-season (regular season only), with per-100 rates,
  computed DRE, and rank. Also prints the final scaled formula.

Usage:
  python scripts/dre/build_wnba_dre.py
  python scripts/dre/build_wnba_dre.py --min-poss 150   # regression sample floor
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

WEHOOP_BOX_URL = ("https://github.com/sportsdataverse/wehoop-wnba-data/raw/main/"
                   "wnba/player_box/parquet/player_box_{season}.parquet")

DASH_V6   = Path("data/wnba_rapm_dashboard_v6.csv")
OUT_CSV   = Path("data/dre/wnba_dre.csv")
FORMULA_TXT = Path("data/dre/wnba_dre_formula.txt")
COEF_JSON = Path("data/dre/wnba_dre_coefficients.json")

SEASONS = list(range(2009, 2027))   # matches our RAPM coverage
TARGET_WINDOW = 2                   # RAPM window (years) used as regression target.
                                     # Validated by scripts/dre/compare_dre_targets.py:
                                     # 2Y maximizes OUT-OF-SAMPLE correlation with each
                                     # season's actual 1Y RAPM (r=0.573). In-sample R²
                                     # rises with window length (0.32@1Y → 0.42@5Y) but
                                     # that is a smoothing artifact, not real skill —
                                     # longer windows wash out the season-specific
                                     # signal DRE exists to estimate.
MIN_POSS_DEFAULT = 700              # season possession floor for the REGRESSION sample
                                     # (≈ a rotation player's full-season on-court exposure)

BOX_STAT_COLS = [
    "points", "fg2a", "three_point_field_goals_attempted", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "assists", "steals", "blocks",
    "turnovers", "fouls",
]
RATE_LABELS = {
    "points": "PTS", "fg2a": "FG2A",
    "three_point_field_goals_attempted": "FG3A",
    "free_throws_attempted": "FTA",
    "offensive_rebounds": "ORB", "defensive_rebounds": "DRB",
    "assists": "AST", "steals": "STL", "blocks": "BLK",
    "turnovers": "TOV", "fouls": "PF",
}


def norm_name(s: str) -> str:
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[.''`\-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── box score loading + possession estimation ──────────────────────────────────

def load_season_box(season: int) -> pd.DataFrame | None:
    url = WEHOOP_BOX_URL.format(season=season)
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return None
        df = pd.read_parquet(io.BytesIO(r.content))
    except Exception:
        return None

    df = df[(df["season_type"] == 2) & (df["did_not_play"] == False)].copy()   # noqa: E712
    df = df[df["minutes"].fillna(0) > 0]
    if df.empty:
        return None

    df["fg2a"] = df["field_goals_attempted"] - df["three_point_field_goals_attempted"]

    # team-game totals (for possession share) — sum ALL players on that team+game
    grp = df.groupby(["game_id", "team_id"])
    team_tot = grp.agg(
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
    # team_min = sum of ALL 5 on-court players' minutes that game (≈200 for a
    # regulation 40-min game, ≈225 with one 5-min OT, etc.) — i.e. 5x the
    # actual game duration, since exactly 5 players are always on the floor.
    # Player's share of team possessions must divide by the GAME DURATION
    # (team_min/5), not by team_min itself, else it's inflated ~5x.
    game_duration = df["team_min"] / 5.0
    df["player_poss"] = np.where(
        game_duration > 0, df["team_poss"] * df["minutes"] / game_duration, 0.0
    )
    df["season"] = season
    return df


def build_player_season_rates(min_poss_floor: float = 0.0) -> pd.DataFrame:
    """One row per (athlete_display_name, season): totals + per-100 rates."""
    all_rows: list[pd.DataFrame] = []
    for s in SEASONS:
        d = load_season_box(s)
        if d is None or d.empty:
            print(f"  {s}: no box data")
            continue
        agg = d.groupby(["athlete_display_name", "season"]).agg(
            minutes=("minutes", "sum"),
            poss=("player_poss", "sum"),
            games=("game_id", "nunique"),
            team=("team_abbreviation", lambda x: x.mode().iloc[0] if len(x) else ""),
            **{RATE_LABELS[c].lower(): (c, "sum") for c in BOX_STAT_COLS},
        ).reset_index()
        all_rows.append(agg)
        print(f"  {s}: {len(agg):,} player-seasons, "
              f"{d['game_id'].nunique()} games", flush=True)

    out = pd.concat(all_rows, ignore_index=True)
    for c in BOX_STAT_COLS:
        lbl = RATE_LABELS[c].lower()
        out[f"{lbl}100"] = np.where(out["poss"] > 0, out[lbl] / out["poss"] * 100.0, 0.0)
    out["norm_name"] = out["athlete_display_name"].apply(norm_name)
    if min_poss_floor:
        out = out[out["poss"] >= min_poss_floor]
    return out


# ── regression ───────────────────────────────────────────────────────────────

def fit_dre(train: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, float, float]:
    """Weighted OLS of RAPM ~ per-100 box rates. Returns (coefs, intercept, r2)."""
    X = train[feature_cols].values
    y = train["rapm"].values
    w = train["poss"].values
    model = LinearRegression()
    model.fit(X, y, sample_weight=w)
    pred = model.predict(X)
    r2 = r2_score(y, pred, sample_weight=w)
    return model.coef_, model.intercept_, r2


def scale_to_pts1(coefs: np.ndarray, intercept: float, feature_cols: list[str]) -> tuple[np.ndarray, float]:
    pts_idx = feature_cols.index("pts100")
    scale = coefs[pts_idx]
    return coefs / scale, intercept / scale


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-poss", type=float, default=MIN_POSS_DEFAULT, dest="min_poss")
    ap.add_argument("--window", type=int, default=TARGET_WINDOW,
                    help="RAPM window (years) used as the regression target")
    ap.add_argument("--output", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    print("Downloading + aggregating WNBA player box scores (2009–2026) …")
    box = build_player_season_rates()
    print(f"\nTotal: {len(box):,} player-seasons across {box['season'].nunique()} seasons")

    print(f"\nLoading RAPM target: v6 {args.window}Y window …")
    rapm = pd.read_csv(DASH_V6)
    rapm = rapm[rapm["rapm_length"] == args.window].copy()
    rapm["norm_name"] = rapm["name"].apply(norm_name)
    rapm = rapm.rename(columns={"end_year": "season"})
    print(f"  {len(rapm):,} rows")

    merged = box.merge(
        rapm[["norm_name", "season", "rapm", "orapm", "drapm", "total_poss"]],
        on=["norm_name", "season"], how="inner",
    )
    print(f"\nMerged (box ∩ RAPM): {len(merged):,} player-seasons")
    unmatched = set(box["norm_name"]) - set(rapm["norm_name"])
    print(f"  ({len(unmatched)} distinct box names never matched any RAPM row — "
          f"mostly deep bench/rare players)")

    feature_cols = [f"{RATE_LABELS[c].lower()}100" for c in BOX_STAT_COLS]

    train = merged[merged["poss"] >= args.min_poss].copy()
    print(f"\nRegression sample (poss ≥ {args.min_poss:.0f}): {len(train):,} player-seasons")

    coefs, intercept, r2 = fit_dre(train, feature_cols)
    scaled_coefs, scaled_intercept = scale_to_pts1(coefs, intercept, feature_cols)

    print(f"\n=== WNBA DRE — weighted regression, R²={r2:.4f} ===")
    print(f"  (weight = season possessions; target = v6 {args.window}Y RAPM)\n")
    labels = [RATE_LABELS[c] for c in BOX_STAT_COLS]

    print("  -- RAW coefficients (this scale approximates RAPM directly) --")
    for lbl, c in zip(labels, coefs):
        print(f"    {lbl:5s}  {c:+.4f}")
    print(f"    intercept  {intercept:+.4f}")

    print("\n  -- SCALED coefficients, PTS=1.00 (Ferrigan's DRE convention; "
          "Game-Score-style bulk metric, NOT on the RAPM scale) --")
    formula_terms = []
    for lbl, c in zip(labels, scaled_coefs):
        sign = "+" if c >= 0 else "-"
        print(f"    {lbl:5s}  {c:+.3f}")
        formula_terms.append(f"{sign} {abs(c):.3f}×{lbl}")
    print(f"    intercept  {scaled_intercept:+.3f}")

    formula_str = "DRE = " + " ".join(formula_terms).lstrip("+ ") + f" {scaled_intercept:+.3f}"
    formula_str = formula_str.replace("+ +", "+").replace("+ -", "- ")
    print(f"\n  {formula_str}")
    print(f"  (all rates are PER 100 POSSESSIONS)")

    FORMULA_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(FORMULA_TXT, "w") as f:
        f.write(f"WNBA DRE — fit on v6 {args.window}Y RAPM, R²={r2:.4f}, "
                f"n={len(train):,} player-seasons (poss≥{args.min_poss:.0f})\n\n")
        f.write("SCALED (PTS=1, DRE convention):\n" + formula_str + "\n\n")
        for lbl, c in zip(labels, scaled_coefs):
            f.write(f"  {lbl:5s}  {c:+.4f}\n")
        f.write(f"  intercept {scaled_intercept:+.4f}\n\n")
        f.write("RAW (approximates RAPM's own scale directly):\n")
        for lbl, c in zip(labels, coefs):
            f.write(f"  {lbl:5s}  {c:+.4f}\n")
        f.write(f"  intercept {intercept:+.4f}\n")
    print(f"\n  Formula saved → {FORMULA_TXT}")

    # Machine-readable coefficients so DRE can be applied to ANY box score
    # later without refitting (see scripts/dre/apply_dre.py).
    import json
    coef_json = {
        "fit_target": f"v6 {args.window}Y RAPM",
        "r2_insample": round(float(r2), 4),
        "n_train": int(len(train)),
        "min_poss_fit": float(args.min_poss),
        "feature_order": [RATE_LABELS[c] for c in BOX_STAT_COLS],
        "scaled": {lbl: round(float(c), 6) for lbl, c in zip(labels, scaled_coefs)},
        "scaled_intercept": round(float(scaled_intercept), 6),
        "raw": {lbl: round(float(c), 6) for lbl, c in zip(labels, coefs)},
        "raw_intercept": round(float(intercept), 6),
    }
    with open(COEF_JSON, "w") as f:
        json.dump(coef_json, f, indent=2)
    print(f"  Coefficients saved → {COEF_JSON}")

    # ── apply to EVERY player-season (not just the regression sample) ────────
    X_all = merged[feature_cols].values
    merged["dre"] = (X_all @ scaled_coefs + scaled_intercept).round(3)          # Ferrigan scale (PTS=1)
    merged["dre_rapm_scale"] = (X_all @ coefs + intercept).round(3)             # directly comparable to RAPM
    merged["dre_total"] = (merged["dre"] * merged["poss"] / 100.0).round(2)

    out_cols = (["athlete_display_name", "team", "season", "games", "minutes", "poss",
                "dre", "dre_rapm_scale", "dre_total", "rapm"]
               + feature_cols)
    out = merged[out_cols].rename(columns={"athlete_display_name": "name"})
    out = out.sort_values(["season", "dre"], ascending=[False, False]).reset_index(drop=True)
    out["dre_rank"] = out.groupby("season")["dre"].rank(ascending=False, method="min").astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out):,} player-seasons → {args.output}")

    yr = out["season"].max()
    print(f"\n=== Top 15 DRE — {yr} ===")
    print(out[out["season"] == yr].head(15)[
        ["name", "team", "poss", "dre", "dre_rapm_scale", "rapm"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
