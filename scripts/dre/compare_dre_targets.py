"""
Which RAPM window should DRE be fit against — 1Y, 2Y, 3Y, 4Y, or 5Y?

We defaulted to 3Y (following the Falkenheim/Nylon-Calculus precedent) but
never validated it on OUR data. This script tests each window empirically.

Two scores, because in-sample R² alone is misleading (a smoother target is
mechanically easier to fit, which would always favor the longest window):

  1. IN-SAMPLE R²  — how well box rates explain that window's RAPM.
     Higher = the target is more "box-score explainable". Biased toward
     long windows, so not decisive on its own.

  2. OUT-OF-SAMPLE r — the real test. Fit DRE weights on seasons < S, then
     ask: how well does the resulting DRE (computed from season S's box
     stats) correlate with season S's ACTUAL NEXT-YEAR 1Y RAPM? This is
     DRE's actual job — estimate real on-court impact from a box line —
     so the window that wins here is the one to use.

Caches box rates to data/dre/wnba_box_rates.csv so reruns are fast.

Usage:
  python scripts/dre/compare_dre_targets.py
  python scripts/dre/compare_dre_targets.py --rebuild-box
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_wnba_dre import (
    build_player_season_rates, norm_name, BOX_STAT_COLS, RATE_LABELS, DASH_V6,
)

BOX_CACHE = Path("data/dre/wnba_box_rates.csv")
OUT_CSV   = Path("data/dre/wnba_dre_target_comparison.csv")
MIN_POSS  = 700
WINDOWS   = [1, 2, 3, 4, 5]


def load_box(rebuild: bool) -> pd.DataFrame:
    if BOX_CACHE.exists() and not rebuild:
        print(f"Loading cached box rates from {BOX_CACHE}")
        return pd.read_csv(BOX_CACHE)
    print("Building box rates from wehoop (this takes a minute) …")
    box = build_player_season_rates()
    BOX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    box.to_csv(BOX_CACHE, index=False)
    print(f"  cached → {BOX_CACHE}")
    return box


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-box", action="store_true", dest="rebuild_box")
    args = ap.parse_args()

    box = load_box(args.rebuild_box)
    box["norm_name"] = box["athlete_display_name"].apply(norm_name)
    feature_cols = [f"{RATE_LABELS[c].lower()}100" for c in BOX_STAT_COLS]

    dash = pd.read_csv(DASH_V6)
    dash["norm_name"] = dash["name"].apply(norm_name)
    dash = dash.rename(columns={"end_year": "season"})

    # Ground truth for out-of-sample scoring: each season's own 1Y RAPM.
    truth = dash[dash["rapm_length"] == 1][["norm_name", "season", "rapm"]].rename(
        columns={"rapm": "rapm_1y_actual"})

    rows = []
    for w in WINDOWS:
        tgt = dash[dash["rapm_length"] == w][["norm_name", "season", "rapm"]]
        m = box.merge(tgt, on=["norm_name", "season"], how="inner")
        m = m[m["poss"] >= MIN_POSS]
        if len(m) < 100:
            continue

        # ── 1. in-sample weighted R² ──────────────────────────────────────
        X, y, sw = m[feature_cols].values, m["rapm"].values, m["poss"].values
        mod = LinearRegression().fit(X, y, sample_weight=sw)
        r2_in = r2_score(y, mod.predict(X), sample_weight=sw)

        # ── 2. out-of-sample: predict each season's ACTUAL 1Y RAPM ────────
        oos_pred, oos_true, oos_w = [], [], []
        seasons = sorted(m["season"].unique())
        for S in seasons[3:]:                       # need history to train on
            tr = m[m["season"] < S]
            te = m[m["season"] == S]
            if len(tr) < 200 or te.empty:
                continue
            mm = LinearRegression().fit(
                tr[feature_cols].values, tr["rapm"].values,
                sample_weight=tr["poss"].values)
            pred = mm.predict(te[feature_cols].values)
            te2 = te[["norm_name", "season", "poss"]].copy()
            te2["pred"] = pred
            te2 = te2.merge(truth, on=["norm_name", "season"], how="inner")
            if te2.empty:
                continue
            oos_pred.extend(te2["pred"]); oos_true.extend(te2["rapm_1y_actual"])
            oos_w.extend(te2["poss"])

        if len(oos_pred) > 50:
            p, t, wt = np.array(oos_pred), np.array(oos_true), np.array(oos_w)
            wm_p, wm_t = np.average(p, weights=wt), np.average(t, weights=wt)
            cov = np.average((p - wm_p) * (t - wm_t), weights=wt)
            r_oos = cov / (np.sqrt(np.average((p-wm_p)**2, weights=wt))
                           * np.sqrt(np.average((t-wm_t)**2, weights=wt)))
            rmse = np.sqrt(np.average((p - t) ** 2, weights=wt))
            n_oos = len(p)
        else:
            r_oos, rmse, n_oos = np.nan, np.nan, 0

        pts_c = mod.coef_[feature_cols.index("pts100")]
        rows.append({
            "window": w, "n_train": len(m),
            "r2_insample": round(r2_in, 4),
            "r_oos_vs_actual_1y": round(r_oos, 4),
            "rmse_oos": round(rmse, 4),
            "n_oos": n_oos,
            "stl_scaled": round(mod.coef_[feature_cols.index("stl100")] / pts_c, 3),
            "tov_scaled": round(mod.coef_[feature_cols.index("tov100")] / pts_c, 3),
            "intercept_scaled": round(mod.intercept_ / pts_c, 2),
        })
        print(f"  {w}Y: n={len(m):,}  R²(in)={r2_in:.4f}  "
              f"r(oos vs actual 1Y)={r_oos:.4f}  rmse={rmse:.3f}", flush=True)

    res = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    print(f"\n=== DRE target comparison (saved → {OUT_CSV}) ===")
    print(res.to_string(index=False))

    best_oos = res.loc[res["r_oos_vs_actual_1y"].idxmax()]
    best_in  = res.loc[res["r2_insample"].idxmax()]
    print(f"\nBest by OUT-OF-SAMPLE r (the decisive test): "
          f"{int(best_oos['window'])}Y  (r={best_oos['r_oos_vs_actual_1y']:.4f})")
    print(f"Best by in-sample R² (biased toward long windows): "
          f"{int(best_in['window'])}Y  (R²={best_in['r2_insample']:.4f})")


if __name__ == "__main__":
    main()
