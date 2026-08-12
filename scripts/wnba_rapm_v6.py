"""
6-Factor RAPM v6 — Jerry Engelmann-style 2N core RAPM.

New vs v5
─────────
  Jerry core      Adapts github.com/jerryengelmann/RAPM/rapm.py:
                    • separate offensive and defensive player variables
                    • defense columns entered as +1 and later negated
                    • target is centered points per possession
                    • alpha grid = 1500..4000 by 250, cv=4
  Possession rows Default input is possession-level data rather than stints,
                  matching the source implementation more closely.

Inherited from v5/v4
─────────────────
  Time decay      Half-life = 700 days, ref = Oct 1 of window end year.
  Second stage    Ridge:  RAPM ≈ β · [oTS, dTS, oTOV, dTOV, oREB, dREB].
  TS% / Reb       Weighted shot denominator, split off/def rebound targets.

Output: data/wnba_rapm_results_v6.csv
  Same player-facing schema as v4, plus core_alpha.

Usage
─────
  python scripts/wnba_rapm_v6.py                   # full run
  python scripts/wnba_rapm_v6.py --end-year 2025   # one end year
  python scripts/wnba_rapm_v6.py --no-decay         # skip time decay
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score

DEFAULT_INPUT   = Path("data/wnba_possessions_all.csv")
DEFAULT_OUTPUT  = Path("data/wnba_rapm_results_v6.csv")
DEFAULT_NAMES   = Path("data/wnba_player_names.csv")
DEFAULT_TEAMS   = Path("data/wnba_team_names.csv")
DEFAULT_DATES   = Path("data/wnba_game_dates.csv")
LIST_DATA_URL   = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt"

CORE_ALPHAS  = [1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000]
FACTOR_ALPHAS = [50, 100, 250, 500, 1000, 2000, 3000, 4000, 8000, 12000]
MIN_POSS     = 50
WINDOW_SIZES = [1, 2, 3, 4, 5]
HALF_LIFE    = 700   # days — matching nbarapm.com

OFF_COLS = [f"offensePlayer{i}Id" for i in range(1, 6)]
DEF_COLS = [f"defensePlayer{i}Id" for i in range(1, 6)]

FACTOR_COLS = [
    "off_ts_rapm", "def_ts_rapm",
    "off_tov_rapm", "def_tov_rapm",
    "off_reb_rapm", "def_reb_rapm",
]


# ── game date lookup ──────────────────────────────────────────────────────────

def _fetch_list() -> dict[str, str]:
    with urlopen(LIST_DATA_URL) as f:
        lines = f.read().decode("utf-8").strip().split("\n")
    return {ln.split("=")[0]: ln.split("=")[1] for ln in lines if "=" in ln}


def _download_tar_csv(key: str, lookup: dict[str, str]) -> pd.DataFrame:
    if key not in lookup:
        return pd.DataFrame()
    url = lookup[key]
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r:
        content = r.read()
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:xz") as tar:
        csv_file = tar.extractfile(f"{key}.csv")
        return pd.read_csv(csv_file)


def build_game_dates(out_path: Path) -> pd.DataFrame:
    """
    Download wnba_shotdetail_* for every season, extract game_id → game_date.
    Saves to out_path.  shotdetail has GAME_DATE as YYYYMMDD integers.
    """
    print("Building game date lookup from shotdetail files …")
    lookup = _fetch_list()
    rows: list[pd.DataFrame] = []
    for season in range(2017, 2026):
        for key in [f"wnba_shotdetail_{season}", f"wnba_shotdetail_po_{season}"]:
            if key not in lookup:
                continue
            print(f"  {key} …", flush=True)
            df = _download_tar_csv(key, lookup)
            if df.empty or "GAME_DATE" not in df.columns:
                continue
            sub = (
                df[["GAME_ID", "GAME_DATE"]]
                .drop_duplicates("GAME_ID")
                .rename(columns={"GAME_ID": "game_id", "GAME_DATE": "game_date"})
            )
            rows.append(sub)

    all_dates = (
        pd.concat(rows, ignore_index=True)
        .drop_duplicates("game_id")
        .copy()
    )
    all_dates["game_date"] = pd.to_datetime(
        all_dates["game_date"].astype(int).astype(str), format="%Y%m%d"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_dates.to_csv(out_path, index=False)
    print(f"  Saved {len(all_dates):,} game dates → {out_path}")
    return all_dates


def load_game_dates(dates_path: Path, rebuild: bool = False) -> dict[int, pd.Timestamp]:
    """Return {game_id: date}.  Downloads if not cached or rebuild=True."""
    if rebuild or not dates_path.exists():
        df = build_game_dates(dates_path)
    else:
        df = pd.read_csv(dates_path, parse_dates=["game_date"])
    return dict(zip(df["game_id"].astype(int), df["game_date"]))


# ── matrix / counting helpers ─────────────────────────────────────────────────

def build_player_index(df: pd.DataFrame) -> dict[int, int]:
    all_ids = pd.concat([df[c] for c in OFF_COLS + DEF_COLS])
    players = sorted(int(x) for x in all_ids.unique() if x != 0)
    return {pid: i for i, pid in enumerate(players)}


def _map_pids(col_vals: np.ndarray, player_idx: dict[int, int]) -> np.ndarray:
    return np.array([player_idx.get(int(p), -1) for p in col_vals])


def build_matrix(
    df: pd.DataFrame,
    player_idx: dict[int, int],
    off_val: float = 1.0,
    def_val: float = -1.0,
    row_mask: np.ndarray | None = None,
) -> csc_matrix:
    if row_mask is not None:
        df = df.iloc[np.where(row_mask)[0]].copy()
    n_rows = len(df)
    n_cols = len(player_idx)
    r_acc, c_acc, d_acc = [], [], []
    row_idx = np.arange(n_rows)
    for cols, val in [(OFF_COLS, off_val), (DEF_COLS, def_val)]:
        if val == 0.0:
            continue
        for c in cols:
            pids  = df[c].values.astype(int)
            cidxs = _map_pids(pids, player_idx)
            mask  = (pids != 0) & (cidxs >= 0)
            r_acc.append(row_idx[mask])
            c_acc.append(cidxs[mask])
            d_acc.append(np.full(mask.sum(), val))
    if not r_acc:
        return csc_matrix((n_rows, n_cols))
    return csc_matrix(
        (np.concatenate(d_acc), (np.concatenate(r_acc), np.concatenate(c_acc))),
        shape=(n_rows, n_cols),
    )


def build_2n_matrix(
    df: pd.DataFrame,
    player_idx: dict[int, int],
    row_mask: np.ndarray | None = None,
) -> csc_matrix:
    """
    Build the core RAPM design matrix with separate player offense and defense
    columns. Following Jerry Engelmann's example, both offense and defense
    entries are +1; positive defensive impact is reported as -coef[N:].
    """
    if row_mask is not None:
        df = df.iloc[np.where(row_mask)[0]].copy()
    n_rows = len(df)
    n_players = len(player_idx)
    r_acc, c_acc, d_acc = [], [], []
    row_idx = np.arange(n_rows)

    for c in OFF_COLS:
        pids = df[c].values.astype(int)
        cidxs = _map_pids(pids, player_idx)
        mask = (pids != 0) & (cidxs >= 0)
        r_acc.append(row_idx[mask])
        c_acc.append(cidxs[mask])
        d_acc.append(np.ones(mask.sum()))

    for c in DEF_COLS:
        pids = df[c].values.astype(int)
        cidxs = _map_pids(pids, player_idx)
        mask = (pids != 0) & (cidxs >= 0)
        r_acc.append(row_idx[mask])
        c_acc.append(cidxs[mask] + n_players)
        d_acc.append(np.ones(mask.sum()))

    if not r_acc:
        return csc_matrix((n_rows, 2 * n_players))
    return csc_matrix(
        (np.concatenate(d_acc), (np.concatenate(r_acc), np.concatenate(c_acc))),
        shape=(n_rows, 2 * n_players),
    )


def count_poss(
    df: pd.DataFrame,
    player_idx: dict[int, int],
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(player_idx)
    off_p = np.zeros(n)
    def_p = np.zeros(n)
    for c in OFF_COLS:
        pids  = df[c].values.astype(int)
        cidxs = _map_pids(pids, player_idx)
        mask  = (pids != 0) & (cidxs >= 0)
        np.add.at(off_p, cidxs[mask], weights[mask])
    for c in DEF_COLS:
        pids  = df[c].values.astype(int)
        cidxs = _map_pids(pids, player_idx)
        mask  = (pids != 0) & (cidxs >= 0)
        np.add.at(def_p, cidxs[mask], weights[mask])
    return off_p, def_p


def player_teams(
    df: pd.DataFrame,
    player_idx: dict[int, int],
    weights: np.ndarray,
) -> np.ndarray:
    n = len(player_idx)
    team_ids = df["offenseTeamId"].values.astype(int)
    acc: dict[int, dict[int, float]] = {j: {} for j in range(n)}
    for c in OFF_COLS:
        pids  = df[c].values.astype(int)
        cidxs = _map_pids(pids, player_idx)
        mask  = (pids != 0) & (cidxs >= 0)
        for j, tid, w in zip(cidxs[mask], team_ids[mask], weights[mask]):
            acc[j][tid] = acc[j].get(tid, 0.0) + w
    result = np.zeros(n, dtype=int)
    for j, d in acc.items():
        if d:
            result[j] = max(d, key=d.get)
    return result


# ── ridge ──────────────────────────────────────────────────────────────────────

def fit_rapm(
    X: csc_matrix,
    y: np.ndarray,
    w: np.ndarray,
    alphas: list[float] = FACTOR_ALPHAS,
) -> tuple[np.ndarray, float]:
    ridge = RidgeCV(alphas=alphas, cv=5, fit_intercept=True)
    ridge.fit(X, y, sample_weight=w)
    return ridge.coef_, float(ridge.alpha_)


# ── helpers ────────────────────────────────────────────────────────────────────

def _per100(num: np.ndarray, poss: np.ndarray) -> np.ndarray:
    return np.where(poss > 0, num / poss * 100.0, 0.0)


def compute_decay_weights(
    game_ids: np.ndarray,
    date_map: dict[int, pd.Timestamp],
    reference_date: pd.Timestamp,
    half_life: int,
) -> np.ndarray:
    """
    Per-row decay: 2^(-(days_ago / half_life)).
    Games with no known date get weight 1.0 (no penalty).
    """
    out = np.ones(len(game_ids))
    for i, gid in enumerate(game_ids):
        date = date_map.get(int(gid))
        if date is not None:
            days_ago = max(0, (reference_date - date).days)
            out[i] = 2.0 ** (-days_ago / half_life)
    return out


# ── second-stage regression ────────────────────────────────────────────────────

_FALLBACK_COEF = np.array([187.6, 187.6, 1.0, 1.0, 1.0, 1.0])  # fixed-scale fallback


def fit_stage2(
    player_rows: list[dict],
    min_poss_s2: int = 100,
) -> tuple[np.ndarray, float]:
    """
    Ridge regression:  RAPM ≈ β · [oTS, dTS, oTOV, dTOV, oREB, dREB]

    Fit on qualified players (>= min_poss_s2 possessions), weighted by
    total_poss.  No intercept — factors are already mean-zero by construction
    of the RAPM ridge.

    Returns (coef [6], weighted_r2).
    Falls back to fixed multipliers if fewer than 10 qualifying players.
    """
    qualified = [r for r in player_rows if r["total_poss"] >= min_poss_s2]
    if len(qualified) < 10:
        return _FALLBACK_COEF.copy(), 0.0

    X = np.array([[r[c] for c in FACTOR_COLS] for r in qualified])
    y = np.array([r["RAPM"] for r in qualified])
    w = np.array([r["total_poss"] for r in qualified])

    model = Ridge(alpha=0.1, fit_intercept=False)
    model.fit(X, y, sample_weight=w)

    y_hat = model.predict(X)
    r2    = float(r2_score(y, y_hat, sample_weight=w))
    return model.coef_, r2


# ── single window ──────────────────────────────────────────────────────────────

def compute_window(
    df: pd.DataFrame,
    min_poss: int,
    date_map: dict[int, pd.Timestamp],
    use_decay: bool,
    end_year: int,
    label: str = "",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    poss   = df["possessions"].values.astype(float)
    poss_w = np.where(poss > 0, poss, 1.0)

    # ── time decay ────────────────────────────────────────────────────────────
    if use_decay and date_map:
        ref_date = pd.Timestamp(f"{end_year}-10-01")
        decay_w  = compute_decay_weights(df["gameId"].values, date_map, ref_date, HALF_LIFE)
        poss_w   = poss_w * decay_w
    else:
        decay_w = np.ones(len(df))

    # ── targets ───────────────────────────────────────────────────────────────
    pts_poss = np.where(poss > 0, df["points"].values.astype(float) / poss, 0.0)
    pts_centered = pts_poss - np.average(pts_poss, weights=poss_w)
    pts100  = _per100(df["points"].values.astype(float), poss)
    tov100  = _per100(df["turnovers"].values.astype(float), poss)
    oreb100 = _per100(df["off_reb"].values.astype(float), poss)
    dreb100 = _per100(df["def_reb"].values.astype(float), poss)

    # TS%: weight by shot denominator (+ decay); drop no-shot rows
    ts_denom = 2.0 * (df["fga"].values + 0.44 * df["fta"].values)
    ts_mask  = ts_denom > 0
    ts_rate  = np.where(ts_mask, df["points"].values / np.where(ts_mask, ts_denom, 1.0), 0.0)
    ts_w     = ts_denom * decay_w   # combine shot volume + recency

    player_idx = build_player_index(df)
    n          = len(player_idx)
    idx_to_pid = {v: k for k, v in player_idx.items()}
    print(f"  {label}: {n} players, {len(df):,} poss ({ts_mask.sum():,} with shots)",
          flush=True)

    # ── design matrices ───────────────────────────────────────────────────────
    X_core   = build_2n_matrix(df, player_idx)
    X_off    = build_matrix(df, player_idx,  1.0,  0.0)
    X_def    = build_matrix(df, player_idx,  0.0,  1.0)
    X_off_ts = build_matrix(df, player_idx,  1.0,  0.0, row_mask=ts_mask)
    X_def_ts = build_matrix(df, player_idx,  0.0,  1.0, row_mask=ts_mask)

    # ── factor regressions ────────────────────────────────────────────────────
    coefs: dict[str, np.ndarray] = {}
    alphas_used: dict[str, float] = {}
    coefs["core"], alphas_used["core"] = fit_rapm(X_core, pts_centered, poss_w, CORE_ALPHAS)
    coefs["ts_off"], alphas_used["ts_off"] = fit_rapm(X_off_ts, ts_rate[ts_mask], ts_w[ts_mask])
    coefs["ts_def"], alphas_used["ts_def"] = fit_rapm(X_def_ts, ts_rate[ts_mask], ts_w[ts_mask])
    coefs["tov_off"], alphas_used["tov_off"] = fit_rapm(X_off, tov100, poss_w)
    coefs["tov_def"], alphas_used["tov_def"] = fit_rapm(X_def, tov100, poss_w)
    coefs["oreb"], alphas_used["oreb"] = fit_rapm(X_off, oreb100, poss_w)
    coefs["dreb"], alphas_used["dreb"] = fit_rapm(X_def, dreb100, poss_w)
    core_orapm = coefs["core"][:n]
    core_drapm = -coefs["core"][n:]

    off_p, def_p = count_poss(df, player_idx, poss_w)
    team_arr     = player_teams(df, player_idx, poss_w)

    # ── build per-player rows (raw factors) ───────────────────────────────────
    rows: list[dict] = []
    for j in range(n):
        op, dp = off_p[j], def_p[j]
        if op + dp < min_poss:
            continue

        orapm = float(core_orapm[j]) * 100.0
        drapm = float(core_drapm[j]) * 100.0
        rapm = orapm + drapm

        rows.append({
            "player_id":    idx_to_pid[j],
            "team_id":      int(team_arr[j]),
            "off_poss":     round(op, 1),
            "def_poss":     round(dp, 1),
            "total_poss":   round(op + dp, 1),
            "ORAPM":        round(orapm, 2),
            "DRAPM":        round(drapm, 2),
            "RAPM":         round(rapm, 2),
            "core_alpha":   alphas_used["core"],
            "off_ts_rapm":  round(float(coefs["ts_off"][j]),    4),
            "def_ts_rapm":  round(-float(coefs["ts_def"][j]),   4),
            "off_tov_rapm": round(-float(coefs["tov_off"][j]),  4),
            "def_tov_rapm": round(float(coefs["tov_def"][j]),   4),
            "off_reb_rapm": round(float(coefs["oreb"][j]),      4),
            "def_reb_rapm": round(-float(coefs["dreb"][j]),     4),
        })

    # ── second-stage: learn β weights, compute *_pts columns ─────────────────
    beta, r2 = fit_stage2(rows)
    print(f"    core α={alphas_used['core']:.0f}  "
          f"stage2 R²={r2:.3f}  "
          f"β_ts={beta[0]:.1f}/{beta[1]:.1f}  "
          f"β_tov={beta[2]:.1f}/{beta[3]:.1f}  "
          f"β_reb={beta[4]:.1f}/{beta[5]:.1f}",
          flush=True)

    for row in rows:
        row["off_ts_pts"]  = round(beta[0] * row["off_ts_rapm"],  2)
        row["def_ts_pts"]  = round(beta[1] * row["def_ts_rapm"],  2)
        row["off_tov_pts"] = round(beta[2] * row["off_tov_rapm"], 2)
        row["def_tov_pts"] = round(beta[3] * row["def_tov_rapm"], 2)
        row["off_reb_pts"] = round(beta[4] * row["off_reb_rapm"], 2)
        row["def_reb_pts"] = round(beta[5] * row["def_reb_rapm"], 2)
        row["stage2_r2"]   = round(r2, 4)

    return pd.DataFrame(rows)


# ── all windows × all end years ───────────────────────────────────────────────

def run_all(
    all_df: pd.DataFrame,
    names_df: pd.DataFrame | None,
    teams_df: pd.DataFrame | None,
    date_map: dict[int, pd.Timestamp],
    output_path: Path,
    min_poss: int,
    window_sizes: list[int],
    only_end_year: int | None,
    use_decay: bool,
) -> pd.DataFrame:

    seasons = sorted(all_df["season"].unique().astype(int))
    print(f"Seasons available: {seasons}")
    print(f"Time decay: {'ON (half-life=' + str(HALF_LIFE) + ' days)' if use_decay else 'OFF'}")
    if use_decay:
        coverage = sum(1 for gid in all_df["gameId"].unique() if int(gid) in date_map)
        pct = coverage / all_df["gameId"].nunique() * 100
        print(f"Date coverage: {coverage}/{all_df['gameId'].nunique()} games ({pct:.1f}%)")

    # Name / team lookups
    name_map: dict[int, tuple[str, str, str]] = {}
    if names_df is not None:
        for _, r in names_df.iterrows():
            full = str(r.get("name") or "")
            last = str(r.get("name_last") or "")
            ini  = str(r.get("name_i") or "")
            if not full and last:
                full = last
            name_map[int(r["player_id"])] = (full, last, ini)

    team_map: dict[int, str] = {}
    if teams_df is not None:
        for _, r in teams_df.iterrows():
            team_map[int(r["team_id"])] = str(r["tricode"])

    all_rows: list[dict] = []

    for w in window_sizes:
        for end_year in seasons:
            if only_end_year and end_year != only_end_year:
                continue
            start = end_year - w + 1
            if start < seasons[0]:
                continue
            window_seasons = [s for s in seasons if start <= s <= end_year]
            if len(window_seasons) < w:
                continue

            df_w  = all_df[all_df["season"].isin(window_seasons)].copy()
            span  = f"{start}–{end_year}" if start != end_year else str(end_year)
            label = f"{w}Y {span}"

            res = compute_window(df_w, min_poss, date_map, use_decay, end_year, label)
            if res.empty:
                continue

            for _, row in res.iterrows():
                pid = int(row["player_id"])
                nm  = name_map.get(pid, ("", "", ""))
                all_rows.append({
                    "player_id":    pid,
                    "name":         nm[0],
                    "name_last":    nm[1],
                    "name_i":       nm[2],
                    "team":         team_map.get(int(row["team_id"]), str(int(row["team_id"]))),
                    "end_year":     end_year,
                    "window":       f"{w}Y",
                    "seasons":      span,
                    "total_poss":   row["total_poss"],
                    "off_poss":     row["off_poss"],
                    "def_poss":     row["def_poss"],
                    "ORAPM":        row["ORAPM"],
                    "DRAPM":        row["DRAPM"],
                    "RAPM":         row["RAPM"],
                    "core_alpha":   row["core_alpha"],
                    "off_ts_rapm":  row["off_ts_rapm"],
                    "def_ts_rapm":  row["def_ts_rapm"],
                    "off_tov_rapm": row["off_tov_rapm"],
                    "def_tov_rapm": row["def_tov_rapm"],
                    "off_reb_rapm": row["off_reb_rapm"],
                    "def_reb_rapm": row["def_reb_rapm"],
                    "off_ts_pts":   row["off_ts_pts"],
                    "def_ts_pts":   row["def_ts_pts"],
                    "off_tov_pts":  row["off_tov_pts"],
                    "def_tov_pts":  row["def_tov_pts"],
                    "off_reb_pts":  row["off_reb_pts"],
                    "def_reb_pts":  row["def_reb_pts"],
                    "stage2_r2":    row["stage2_r2"],
                })

    combined = (
        pd.DataFrame(all_rows)
        .sort_values(["end_year", "window", "RAPM"], ascending=[False, True, False])
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\nSaved {len(combined):,} rows → {output_path}")

    preview = combined[
        (combined["window"] == "5Y") &
        (combined["end_year"] == combined["end_year"].max()) &
        (combined["total_poss"] >= 500)
    ]
    print(preview[[
        "name", "team", "total_poss", "ORAPM", "DRAPM", "RAPM",
        "off_ts_pts", "def_ts_pts", "off_reb_pts", "def_reb_pts", "stage2_r2"
    ]].head(20).to_string(index=False))
    return combined


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WNBA 6-Factor RAPM v6 — Jerry Engelmann-style 2N core RAPM"
    )
    p.add_argument("--input",          type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output",         type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--names",          type=Path, default=DEFAULT_NAMES)
    p.add_argument("--teams",          type=Path, default=DEFAULT_TEAMS)
    p.add_argument("--dates",          type=Path, default=DEFAULT_DATES)
    p.add_argument("--end-year",       type=int,  default=None, dest="end_year")
    p.add_argument("--min-poss",       type=int,  default=MIN_POSS, dest="min_poss")
    p.add_argument("--windows",        type=int,  nargs="+", default=WINDOW_SIZES)
    p.add_argument("--no-decay",       action="store_true", dest="no_decay",
                   help="Disable time decay (equal weights across window)")
    p.add_argument("--rebuild-dates",  action="store_true", dest="rebuild_dates",
                   help="Re-download game dates from shotdetail even if cached")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    use_decay = not args.no_decay

    # Load possession data
    print(f"Loading {args.input} …")
    df = pd.read_csv(args.input)
    if "lineup_complete" in df.columns:
        before = len(df)
        df = df[df["lineup_complete"] == 1]
        print(f"  {before:,} → {len(df):,} after lineup_complete filter")
    for c in OFF_COLS + DEF_COLS:
        df[c] = df[c].fillna(0).astype(int)

    # Load supporting data
    names_df = pd.read_csv(args.names) if args.names.exists() else None
    teams_df = pd.read_csv(args.teams) if args.teams.exists() else None
    if names_df is not None:
        print(f"  {len(names_df):,} player names loaded")
    if teams_df is not None:
        print(f"  {len(teams_df):,} team tricodes loaded")

    # Load game dates (for time decay)
    if use_decay:
        date_map = load_game_dates(args.dates, rebuild=args.rebuild_dates)
        print(f"  {len(date_map):,} game dates loaded")
    else:
        date_map = {}

    run_all(
        df, names_df, teams_df, date_map,
        args.output, args.min_poss, args.windows,
        args.end_year, use_decay,
    )
