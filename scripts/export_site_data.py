"""
Export the RAPM CSVs to the compact JSON the static site consumes.

Reads (relative to --repo-root):
  data/wnba_rapm_dashboard_v6.csv      windowed RAPM, the main view
  data/td_rapm/wnba_rapm_td.csv        career time-decayed RAPM

Writes:
  site/data/players.json  {"cols":[...], "rows":[[...], ...]}
  site/data/td.json       same shape
  site/data/meta.json     {"updated","seasons","windows","td_as_of"}

Design notes:
  * Column-array form (cols + rows) rather than a list of objects: repeating 17
    key names across 17k rows roughly triples the payload for no benefit, since
    app.js indexes columns positionally anyway.
  * The precomputed *_rank columns are deliberately dropped. The client
    recomputes ranks and percentiles over the *filtered* pool, so a rank
    baked in against the full-cohort pool would disagree with what the user
    sees the moment they touch a filter.
  * Every input is optional. A missing CSV emits a valid empty payload with
    the expected header, so the site renders an empty table instead of 404ing
    and taking the whole page down with it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# --- output schemas -------------------------------------------------------
# These double as the header for the empty-payload fallback, so the site can
# still build a table head when an input CSV is missing.

PLAYERS_COLS = [
    "name", "team", "end_year", "rapm_length", "total_poss",
    "orapm", "rapm", "drapm",
    "off_ts_val", "off_tov_val", "off_reb_val",
    "def_ts_val", "def_tov_val", "def_reb_val",
    "o_poss_val", "d_poss_val", "poss_val",
]

TD_COLS = [
    "name", "team", "total_poss",
    "ORAPM", "RAPM", "DRAPM",
    "off_ts_pts", "off_tov_pts", "off_reb_pts",
    "def_ts_pts", "def_tov_pts", "def_reb_pts",
]

NDIGITS = 3
DEFAULT_WINDOWS = [1, 2, 3, 4, 5]


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def load_csv(path: Path, label: str) -> pd.DataFrame | None:
    """Read a CSV, returning None (with a warning) if it is missing or unreadable."""
    if not path.exists():
        warn(f"{label}: {path} not found - emitting empty payload")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # malformed/truncated file shouldn't kill the build
        warn(f"{label}: failed to read {path} ({exc}) - emitting empty payload")
        return None
    if df.empty:
        warn(f"{label}: {path} has no rows")
    return df


def clean_column(series: pd.Series) -> list:
    """One column -> JSON-ready Python scalars.

    Ints stay ints, floats round to <=3 decimals, NaN/NaT become None. Float
    columns whose values are all integral are emitted as ints: it shrinks the
    payload and JS makes no distinction between 735 and 735.0 anyway.
    """
    if pd.api.types.is_bool_dtype(series):
        return [None if pd.isna(v) else bool(v) for v in series]

    if pd.api.types.is_integer_dtype(series):
        # A pandas nullable Int64 can still hold NA.
        return [None if pd.isna(v) else int(v) for v in series]

    if pd.api.types.is_float_dtype(series):
        vals = series.to_numpy(dtype="float64", copy=False)
        finite = vals[~pd.isna(vals)]
        integral = bool(len(finite)) and bool((finite == finite.round()).all())
        out = []
        for v in vals:
            if math.isnan(v) or math.isinf(v):
                out.append(None)
            elif integral:
                out.append(int(v))
            else:
                r = round(float(v), NDIGITS)
                # Collapse -0.0 so the site never renders a signed negative zero.
                out.append(0.0 if r == 0 else r)
        return out

    # Object/string: preserve text, blanks become null.
    out = []
    for v in series:
        if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
            out.append(None)
        else:
            s = str(v).strip()
            out.append(s if s else None)
    return out


def build_payload(df: pd.DataFrame | None, cols: list[str], label: str) -> dict:
    """Project `df` down to `cols` and emit the {"cols","rows"} contract."""
    if df is None:
        return {"cols": list(cols), "rows": []}

    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        warn(f"{label}: input is missing expected column(s) {missing} - omitted from output")

    if not present:
        warn(f"{label}: none of the expected columns are present - emitting empty payload")
        return {"cols": list(cols), "rows": []}

    columns = [clean_column(df[c]) for c in present]
    rows = [list(r) for r in zip(*columns)]
    return {"cols": present, "rows": rows}


def write_json(path: Path, obj: dict, *, compact: bool) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    else:
        text = json.dumps(obj, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path.stat().st_size


def human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def build_meta(players: pd.DataFrame | None, td: pd.DataFrame | None) -> dict:
    seasons: list[int] = []
    windows: list[int] = []
    if players is not None:
        if "end_year" in players.columns:
            seasons = sorted({int(v) for v in players["end_year"].dropna()})
        if "rapm_length" in players.columns:
            windows = sorted({int(v) for v in players["rapm_length"].dropna()})

    if not windows:
        # Keep the 1Y..5Y select populated even if the main CSV is missing.
        windows = list(DEFAULT_WINDOWS)

    td_as_of = None
    if td is not None and "as_of" in td.columns:
        vals = sorted({str(v) for v in td["as_of"].dropna()})
        if len(vals) > 1:
            warn(f"td: multiple as_of values {vals} - using the most recent")
        td_as_of = vals[-1] if vals else None

    return {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "seasons": seasons,
        "windows": windows,
        "td_as_of": td_as_of,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export WNBA RAPM CSVs to site/data/*.json for the static site."
    )
    ap.add_argument("--repo-root", default=".", help="repo root (default: .)")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    outdir = repo / "site" / "data"

    players = load_csv(repo / "data" / "wnba_rapm_dashboard_v6.csv", "players")
    td = load_csv(repo / "data" / "td_rapm" / "wnba_rapm_td.csv", "td")

    # A blank player name renders as an empty row in the leaderboard, so surface
    # it here rather than letting it quietly ship.
    if players is not None and "name" in players.columns:
        blank = int(players["name"].isna().sum() + (players["name"].astype(str).str.strip() == "").sum())
        if blank:
            warn(f"players: {blank} row(s) have a blank name - exported as null")

    payloads = [
        ("players.json", build_payload(players, PLAYERS_COLS, "players")),
        ("td.json", build_payload(td, TD_COLS, "td")),
    ]

    print(f"repo root : {repo}")
    print(f"output    : {outdir}")
    print()

    total = 0
    for fname, payload in payloads:
        size = write_json(outdir / fname, payload, compact=True)
        total += size
        print(f"  {fname:<14} {len(payload['rows']):>7,} rows  {len(payload['cols']):>3} cols  {human_size(size):>10}")

    meta = build_meta(players, td)
    size = write_json(outdir / "meta.json", meta, compact=False)
    total += size
    print(f"  {'meta.json':<14} {'':>7}       {'':>3}       {human_size(size):>10}")
    print(f"\n  total {human_size(total)}")
    print(f"  seasons  {meta['seasons'][0] if meta['seasons'] else '-'}"
          f"..{meta['seasons'][-1] if meta['seasons'] else '-'}"
          f"  windows {meta['windows']}  td_as_of {meta['td_as_of']}")
    print(f"  updated  {meta['updated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
