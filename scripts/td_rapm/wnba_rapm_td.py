"""
Career-decay Time-Decay RAPM (DARKO-style) for the WNBA.

Unlike the windowed v6 (1Y–5Y buckets), this uses a player's ENTIRE history
(2009–2026) with an exponential time-decay β^days_ago as the only weighting —
no arbitrary "last X seasons" cutoff. It produces one snapshot per player as of
a chosen reference date, daily-updatable.

  weight_i = possessions_i · β^(days_ago_i),   β = 2^(-1/HALF_LIFE)

HALF_LIFE = 700 days (β ≈ 0.9990) is the empirically optimal decay for WNBA,
confirmed by scripts/optimize_td_rapm_beta.py (peak at ~693 days; far slower
than DARKO's NBA β≈0.99 because WNBA samples are ~1/6 the size).

Model (identical to v6):
  • Core: 2N +1/+1 ridge (Jerry Engelmann), target = centered pts/poss × 100.
    ORAPM = coef[:N]·100,  DRAPM = -coef[N:]·100,  RAPM = ORAPM + DRAPM.
  • Six one-sided factor regressions: oTS, dTS, oTOV, dTOV, oREB, dREB.
  • Second stage: Ridge RAPM ≈ β·[6 factors] → *_pts columns (pts/100).

Decay dates:
  • Real game dates for 2017–2026 (2017–2025 from game_dates.csv, 2026 ESPN).
  • Season-midpoint approximation (Jul 15) for 2009–2016 (no daily dates).
    (NOTE: v6's compute_decay_weights defaults unknown dates to weight 1.0,
     which is wrong for a career model — we approximate instead.)

Output: data/wnba_rapm_td.csv  (one row per qualified player, ranked)

Usage:
  python scripts/wnba_rapm_td.py                     # as-of latest game date
  python scripts/wnba_rapm_td.py --as-of 2025-09-15  # historical snapshot
  python scripts/wnba_rapm_td.py --half-life 1000    # try slower decay
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ (for wnba_rapm_v6)
from wnba_rapm_v6 import (
    build_player_index, build_2n_matrix, build_matrix, fit_rapm,
    count_poss, player_teams, fit_stage2, _per100,
    CORE_ALPHAS, OFF_COLS, DEF_COLS,
)

# Paths are relative to the project root (run from there); inputs are shared
# with the rest of the pipeline, TD outputs live in data/td_rapm/.
POSS_CSV   = Path("data/wnba_possessions_all.csv")
DATES_CSV  = Path("data/wnba_game_dates.csv")
NAMES_CSV  = Path("data/wnba_player_names.csv")
TEAMS_CSV  = Path("data/wnba_team_names.csv")
OUT_CSV    = Path("data/td_rapm/wnba_rapm_td.csv")
WEHOOP_URL = ("https://github.com/sportsdataverse/wehoop-wnba-data/raw/main/"
              "wnba/pbp/parquet/play_by_play_2026.parquet")

HALF_LIFE = 700      # days; β = 2^(-1/HALF_LIFE) ≈ 0.9990 (validated optimum)
MIN_POSS  = 50       # minimum DECAYED possessions to qualify


def load_date_map() -> dict[int, pd.Timestamp]:
    dmap: dict[int, pd.Timestamp] = {}
    if DATES_CSV.exists():
        d = pd.read_csv(DATES_CSV, parse_dates=["game_date"])
        dmap.update(dict(zip(d["game_id"].astype(int), d["game_date"])))
    try:
        pbp = pd.read_parquet(io.BytesIO(requests.get(WEHOOP_URL, timeout=60).content))
        gd = pbp[["game_id", "game_date"]].dropna().drop_duplicates("game_id")
        for gid, dt in zip(gd["game_id"].astype(int), pd.to_datetime(gd["game_date"])):
            dmap.setdefault(int(gid), dt)
    except Exception as e:
        print(f"  (2026 dates unavailable: {e})")
    return dmap


def most_recent_team(df: pd.DataFrame, date_map: dict[int, pd.Timestamp],
                     ref: pd.Timestamp) -> dict[int, int]:
    """
    Map each player → the team of their MOST RECENT game on/before `ref`.

    The career-weighted "most common team" (v6's player_teams) returns a
    player's historical home (e.g. Alyssa Thomas → CON), which is wrong for a
    current snapshot. Using the latest game gives the team they play for NOW.
    """
    d = df.copy()
    d["_date"] = [date_map.get(int(g)) or pd.Timestamp(f"{int(s)}-07-15")
                  for g, s in zip(d["gameId"], d["season"])]
    d = d[d["_date"] <= ref]
    off = d.melt(id_vars=["_date", "offenseTeamId"], value_vars=OFF_COLS,
                 value_name="pid").rename(columns={"offenseTeamId": "team"})[["_date", "team", "pid"]]
    def_ = d.melt(id_vars=["_date", "defenseTeamId"], value_vars=DEF_COLS,
                  value_name="pid").rename(columns={"defenseTeamId": "team"})[["_date", "team", "pid"]]
    long = pd.concat([off, def_], ignore_index=True)
    long = long[long["pid"] != 0]
    latest = long.loc[long.groupby("pid")["_date"].idxmax()]
    return dict(zip(latest["pid"].astype(int), latest["team"].astype(int)))


def decay_weights(game_ids, seasons, date_map, ref_date, half_life) -> np.ndarray:
    """possessions-independent β^days_ago; season-midpoint for unknown dates."""
    out = np.empty(len(game_ids))
    ln_beta = -math.log(2) / half_life
    for i, (gid, s) in enumerate(zip(game_ids, seasons)):
        dt = date_map.get(int(gid)) or pd.Timestamp(f"{int(s)}-07-15")
        days = max(0, (ref_date - dt).days)
        out[i] = math.exp(ln_beta * days)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", type=str, default=None, dest="as_of",
                    help="Reference date YYYY-MM-DD (default: latest game date)")
    ap.add_argument("--half-life", type=int, default=HALF_LIFE, dest="half_life")
    ap.add_argument("--min-poss", type=int, default=MIN_POSS, dest="min_poss")
    ap.add_argument("--output", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    beta = 2 ** (-1 / args.half_life)
    print(f"Loading {POSS_CSV} …")
    df = pd.read_csv(POSS_CSV)
    df = df[df["lineup_complete"] == 1].copy()
    for c in OFF_COLS + DEF_COLS:
        df[c] = df[c].fillna(0).astype(int)
    print(f"  {len(df):,} complete-lineup possessions, "
          f"{df['season'].min()}–{df['season'].max()}")

    date_map = load_date_map()

    # reference date
    if args.as_of:
        ref = pd.Timestamp(args.as_of)
    else:
        known = [date_map[int(g)] for g in df["gameId"].unique() if int(g) in date_map]
        ref = max(known) if known else pd.Timestamp.today()
    print(f"  Half-life {args.half_life}d (β={beta:.5f}), as-of {ref.date()}")

    # ── decay + targets ───────────────────────────────────────────────────────
    poss   = df["possessions"].values.astype(float)
    poss_w = np.where(poss > 0, poss, 1.0)
    decay  = decay_weights(df["gameId"].values, df["season"].values,
                           date_map, ref, args.half_life)
    poss_w = poss_w * decay

    pts_poss     = np.where(poss > 0, df["points"].values / poss, 0.0)
    pts_centered = pts_poss - np.average(pts_poss, weights=poss_w)
    tov100  = _per100(df["turnovers"].values.astype(float), poss)
    oreb100 = _per100(df["off_reb"].values.astype(float), poss)
    dreb100 = _per100(df["def_reb"].values.astype(float), poss)

    ts_denom = 2.0 * (df["fga"].values + 0.44 * df["fta"].values)
    ts_mask  = ts_denom > 0
    ts_rate  = np.where(ts_mask, df["points"].values / np.where(ts_mask, ts_denom, 1.0), 0.0)
    ts_w     = ts_denom * decay

    pidx = build_player_index(df)
    n    = len(pidx)
    idx_to_pid = {v: k for k, v in pidx.items()}
    print(f"  {n} players in design matrix")

    # ── design matrices ───────────────────────────────────────────────────────
    X_core   = build_2n_matrix(df, pidx)
    X_off    = build_matrix(df, pidx, 1.0, 0.0)
    X_def    = build_matrix(df, pidx, 0.0, 1.0)
    X_off_ts = build_matrix(df, pidx, 1.0, 0.0, row_mask=ts_mask)
    X_def_ts = build_matrix(df, pidx, 0.0, 1.0, row_mask=ts_mask)

    print("  Fitting core + 6 factor regressions …")
    core, a_core = fit_rapm(X_core, pts_centered, poss_w, CORE_ALPHAS)
    ts_off, _ = fit_rapm(X_off_ts, ts_rate[ts_mask], ts_w[ts_mask])
    ts_def, _ = fit_rapm(X_def_ts, ts_rate[ts_mask], ts_w[ts_mask])
    tov_off, _ = fit_rapm(X_off, tov100, poss_w)
    tov_def, _ = fit_rapm(X_def, tov100, poss_w)
    oreb, _ = fit_rapm(X_off, oreb100, poss_w)
    dreb, _ = fit_rapm(X_def, dreb100, poss_w)

    core_orapm = core[:n] * 100.0
    core_drapm = -core[n:] * 100.0

    off_p, def_p = count_poss(df, pidx, poss_w)
    raw_off, raw_def = count_poss(df, pidx, np.where(poss > 0, poss, 1.0))
    recent_team = most_recent_team(df, date_map, ref)   # current team, not career home

    rows: list[dict] = []
    for j in range(n):
        if off_p[j] + def_p[j] < args.min_poss:
            continue
        orapm, drapm = float(core_orapm[j]), float(core_drapm[j])
        rows.append({
            "player_id":    idx_to_pid[j],
            "team_id":      int(recent_team.get(idx_to_pid[j], 0)),
            "total_poss":   round(off_p[j] + def_p[j], 1),       # decayed/effective
            "total_poss_raw": int(raw_off[j] + raw_def[j]),
            "ORAPM": round(orapm, 2), "DRAPM": round(drapm, 2),
            "RAPM":  round(orapm + drapm, 2),
            "off_ts_rapm":  round(float(ts_off[j]),   4),
            "def_ts_rapm":  round(-float(ts_def[j]),  4),
            "off_tov_rapm": round(-float(tov_off[j]), 4),
            "def_tov_rapm": round(float(tov_def[j]),  4),
            "off_reb_rapm": round(float(oreb[j]),     4),
            "def_reb_rapm": round(-float(dreb[j]),    4),
        })

    bs, r2 = fit_stage2(rows)
    print(f"  core α={a_core:.0f}  stage2 R²={r2:.3f}  "
          f"β_ts={bs[0]:.1f}/{bs[1]:.1f}  β_tov={bs[2]:.1f}/{bs[3]:.1f}  "
          f"β_reb={bs[4]:.1f}/{bs[5]:.1f}")
    for r in rows:
        r["off_ts_pts"]  = round(bs[0] * r["off_ts_rapm"],  2)
        r["def_ts_pts"]  = round(bs[1] * r["def_ts_rapm"],  2)
        r["off_tov_pts"] = round(bs[2] * r["off_tov_rapm"], 2)
        r["def_tov_pts"] = round(bs[3] * r["def_tov_rapm"], 2)
        r["off_reb_pts"] = round(bs[4] * r["off_reb_rapm"], 2)
        r["def_reb_pts"] = round(bs[5] * r["def_reb_rapm"], 2)

    out = pd.DataFrame(rows)

    # names / teams
    if NAMES_CSV.exists():
        nm = pd.read_csv(NAMES_CSV).set_index("player_id")["name"].to_dict()
        out["name"] = out["player_id"].map(nm).fillna("")
    if TEAMS_CSV.exists():
        tm = pd.read_csv(TEAMS_CSV).set_index("team_id")["tricode"].to_dict()
        out["team"] = out["team_id"].map(tm).fillna(out["team_id"].astype(str))

    out["as_of"]    = ref.date()
    out["half_life"] = args.half_life
    out["stage2_r2"] = round(r2, 4)
    for col in ["ORAPM", "DRAPM", "RAPM"]:
        out[f"{col}_rank"] = out[col].rank(ascending=False, method="min").astype(int)

    cols = ["name", "team", "as_of", "half_life", "total_poss", "total_poss_raw",
            "ORAPM", "RAPM", "DRAPM",
            "off_ts_pts", "off_tov_pts", "off_reb_pts",
            "def_ts_pts", "def_tov_pts", "def_reb_pts",
            "off_ts_rapm", "def_ts_rapm", "off_tov_rapm",
            "def_tov_rapm", "off_reb_rapm", "def_reb_rapm",
            "ORAPM_rank", "DRAPM_rank", "RAPM_rank", "stage2_r2"]
    out = out[cols].sort_values("RAPM", ascending=False).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out):,} players → {args.output}\n")
    print(f"=== Top 20 career-decay TD-RAPM (as-of {ref.date()}, HL={args.half_life}d) ===")
    print(out.head(20)[["name", "team", "total_poss", "ORAPM", "DRAPM", "RAPM"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
