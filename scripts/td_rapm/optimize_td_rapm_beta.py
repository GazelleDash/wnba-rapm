"""
Optimize the time-decay β for WNBA RAPM.

Idea:
  Weight every historical possession by β^(days_ago); pick the β that best
  predicts FUTURE results out-of-sample. β∈(0,1); closer to 1 = slower decay.

Our v6 already uses β = 2^(-1/700) ≈ 0.9990 (a 700-day half-life). This script
sweeps a range of β and measures, for each, how well a core RAPM fit on all
history-to-date predicts the NEXT season's possessions.

Method
──────
  For each TEST_YEAR T:
    • train = all possessions in seasons < T
    • test  = possessions in season T   (strictly out-of-sample)
    • reference date = May 1 of T  (≈ start of season T)
    • for each β:
        w_i        = possessions_i * β^(days_ago_i)
        α          = LAMBDA * Σw / 2         (adapts ridge to effective n)
        fit Ridge core RAPM (2N +1/+1 design) on (train, w)
        predict points-per-possession on the test season
        score RMSE / R² vs actual
  Average across test years → best β.

β↔half-life:  half_life_days = -ln(2) / ln(β)

Output: data/wnba_td_beta_validation.csv  (one row per test_year × β)
Prints the averaged leaderboard and the winning β.

Usage:
  python scripts/optimize_td_rapm_beta.py
  python scripts/optimize_td_rapm_beta.py --test-years 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ (for wnba_rapm_v6)
from wnba_rapm_v6 import build_player_index, build_2n_matrix, OFF_COLS, DEF_COLS

POSS_CSV   = Path("data/wnba_possessions_all.csv")
DATES_CSV  = Path("data/wnba_game_dates.csv")
OUT_CSV    = Path("data/td_rapm/wnba_td_beta_validation.csv")
WEHOOP_URL = ("https://github.com/sportsdataverse/wehoop-wnba-data/raw/main/"
              "wnba/pbp/parquet/play_by_play_2026.parquet")

# β grid: from a fast NBA-scale decay to very slow. Includes our current 0.9990.
BETAS = [0.997, 0.998, 0.9985, 0.9990, 0.9993, 0.9995, 0.9997, 1.0]
LAMBDA = 0.05          # α = LAMBDA * Σw / 2  (gives α≈v6's ~2000 range)
TEST_YEARS_DEFAULT = [2019, 2020, 2021, 2022, 2023, 2024, 2025]


def half_life(beta: float) -> float:
    if beta >= 1.0:
        return float("inf")
    return -math.log(2) / math.log(beta)


def load_date_map() -> dict[int, pd.Timestamp]:
    """Real dates 2017–2025 + 2026 (ESPN) + approx season-midpoint for 2009–2016."""
    dmap: dict[int, pd.Timestamp] = {}
    if DATES_CSV.exists():
        d = pd.read_csv(DATES_CSV, parse_dates=["game_date"])
        dmap.update(dict(zip(d["game_id"].astype(int), d["game_date"])))
    # 2026 from ESPN parquet (game_id → game_date)
    try:
        pbp = pd.read_parquet(io.BytesIO(requests.get(WEHOOP_URL, timeout=60).content))
        gd = (pbp[["game_id", "game_date"]].dropna().drop_duplicates("game_id"))
        for gid, dt in zip(gd["game_id"].astype(int), pd.to_datetime(gd["game_date"])):
            dmap.setdefault(int(gid), dt)
    except Exception as e:
        print(f"  (2026 dates unavailable: {e})")
    return dmap


def decay_weights(game_ids, seasons, date_map, ref_date, beta) -> np.ndarray:
    """β^days_ago per row; real date if known, else season-midpoint (Jul 15)."""
    if beta >= 1.0:
        return np.ones(len(game_ids))
    out = np.empty(len(game_ids))
    ln_beta = math.log(beta)
    for i, (gid, s) in enumerate(zip(game_ids, seasons)):
        dt = date_map.get(int(gid))
        if dt is None:
            dt = pd.Timestamp(f"{int(s)}-07-15")        # approx (pre-2017)
        days = max(0, (ref_date - dt).days)
        out[i] = math.exp(ln_beta * days)
    return out


def per_poss_points(df: pd.DataFrame) -> np.ndarray:
    poss = df["possessions"].values.astype(float)
    return np.where(poss > 0, df["points"].values.astype(float) / poss, 0.0)


def run(test_years: list[int]) -> pd.DataFrame:
    print(f"Loading {POSS_CSV} …")
    df = pd.read_csv(POSS_CSV)
    df = df[df["lineup_complete"] == 1].copy()
    for c in OFF_COLS + DEF_COLS:
        df[c] = df[c].fillna(0).astype(int)
    print(f"  {len(df):,} complete-lineup possessions, "
          f"seasons {df['season'].min()}–{df['season'].max()}")

    date_map = load_date_map()
    print(f"  {len(date_map):,} game dates loaded")

    rows: list[dict] = []
    for T in test_years:
        train = df[df["season"] < T]
        test  = df[df["season"] == T]
        if train.empty or test.empty:
            continue
        ref = pd.Timestamp(f"{T}-05-01")
        pidx = build_player_index(train)

        Xtr = build_2n_matrix(train, pidx)
        ytr = per_poss_points(train)
        Xte = build_2n_matrix(test, pidx)         # uses TRAIN index → unseen→0
        yte = per_poss_points(test)

        poss_tr = train["possessions"].values.astype(float)
        days_w  = {}   # cache base decay per β
        print(f"\nTEST {T}: train {len(train):,} poss ({len(pidx)} players) "
              f"→ predict {len(test):,} poss")

        # baseline: predict train weighted mean
        base_pred = np.average(ytr)
        base_rmse = math.sqrt(np.mean((yte - base_pred) ** 2))

        for beta in BETAS:
            dec = decay_weights(train["gameId"].values, train["season"].values,
                                date_map, ref, beta)
            w = poss_tr * dec
            alpha = LAMBDA * w.sum() / 2.0
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(Xtr, ytr, sample_weight=w)
            pred = ridge.predict(Xte)
            rmse = math.sqrt(np.mean((yte - pred) ** 2))
            r2   = r2_score(yte, pred)
            rows.append({
                "test_year": T, "beta": beta,
                "half_life_days": round(half_life(beta), 1),
                "alpha": round(alpha, 0),
                "n_train": len(train), "n_test": len(test),
                "rmse": round(rmse, 5), "r2": round(r2, 6),
                "rmse_vs_baseline": round(base_rmse - rmse, 6),
            })
            print(f"   β={beta:.4f} (HL={half_life(beta):6.0f}d)  "
                  f"α={alpha:6.0f}  RMSE={rmse:.5f}  R²={r2:+.5f}", flush=True)

    res = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(res)} rows → {OUT_CSV}")

    # ── averaged leaderboard ────────────────────────────────────────────────
    print("\n=== Averaged across test years (higher R² / RMSE-gain = better) ===")
    agg = (res.groupby("beta")
              .agg(mean_r2=("r2", "mean"),
                   mean_rmse=("rmse", "mean"),
                   mean_gain=("rmse_vs_baseline", "mean"),
                   mean_hl=("half_life_days", "mean"))
              .reset_index()
              .sort_values("mean_r2", ascending=False))
    for _, r in agg.iterrows():
        star = "  ← BEST" if r["beta"] == agg.iloc[0]["beta"] else ""
        hl = "∞" if math.isinf(r["mean_hl"]) else f"{r['mean_hl']:.0f}d"
        print(f"  β={r['beta']:.4f}  HL={hl:>7}  meanR²={r['mean_r2']:+.5f}  "
              f"meanRMSE={r['mean_rmse']:.5f}{star}")

    best = agg.iloc[0]["beta"]
    print(f"\nBest β = {best:.4f}  (half-life ≈ "
          f"{'∞' if best>=1 else str(round(half_life(best)))+' days'})")
    print(f"Current v6 uses β ≈ 0.9990 (700-day half-life) for comparison.")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-years", type=int, nargs="+", default=TEST_YEARS_DEFAULT)
    args = ap.parse_args()
    run(args.test_years)
