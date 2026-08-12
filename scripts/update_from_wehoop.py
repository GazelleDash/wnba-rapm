"""
One-shot updater: refresh the current WNBA season from the free wehoop data
repo (sportsdataverse/wehoop-wnba-data) and rebuild all downstream outputs.

wehoop-wnba-data publishes the entire in-progress season as a single parquet,
refreshed ~daily. Run this script any time to pull the latest games and
regenerate RAPM. It is idempotent and re-runnable.

Pipeline (each step is the existing canonical script):
  1. build_season_from_espn.py  — pull season parquet, crosswalk IDs, merge into
                                 data/wnba_possessions_all.csv
  2. wnba_rapm_v6.py          — recompute ONLY the current end-year windows
                                 (historical years are unaffected) and splice
                                 them into data/wnba_rapm_results_v6.csv
  3. format_rapm_dashboard.py — refresh data/wnba_rapm_dashboard_v6.csv
  4. td_rapm/wnba_rapm_td.py  — refresh the career-decay snapshot
                                 data/td_rapm/wnba_rapm_td.csv (as-of latest)

Why splice instead of full rebuild:
  A window ending in year Y only uses seasons ≤ Y. New games in the current
  season S change windows ending in S only — every prior end-year row is
  identical. So we recompute end_year=S (~1–2 min) and replace just those rows,
  instead of refitting all 18 seasons (~10 min).

Usage:
  python scripts/update_from_wehoop.py                 # current season (auto)
  python scripts/update_from_wehoop.py --season 2026
  python scripts/update_from_wehoop.py --skip-td       # skip TD snapshot
  python scripts/update_from_wehoop.py --full-rebuild  # refit every year
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
SCRIPTS    = ROOT / "scripts"
RESULTS_V6 = ROOT / "data" / "wnba_rapm_results_v6.csv"
DASH_V6    = ROOT / "data" / "wnba_rapm_dashboard_v6.csv"
TMP_V6     = ROOT / "data" / "_tmp_v6_endyear.csv"
PY = sys.executable


def run(label: str, cmd: list[str]) -> None:
    print(f"\n{'='*70}\n▶ {label}\n{'='*70}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        raise SystemExit(f"✗ Step failed ({label}); aborting.")
    print(f"✓ {label} done in {time.time()-t0:.0f}s", flush=True)


def splice_endyear(season: int) -> None:
    """Replace only end_year==season rows in the full v6 results file."""
    full = pd.read_csv(RESULTS_V6)
    new  = pd.read_csv(TMP_V6)
    before = len(full)
    full = full[full["end_year"] != season]
    combined = (pd.concat([full, new], ignore_index=True)
                  .sort_values(["end_year", "window", "RAPM"],
                               ascending=[False, True, False])
                  .reset_index(drop=True))
    combined.to_csv(RESULTS_V6, index=False)
    TMP_V6.unlink(missing_ok=True)
    print(f"  spliced end_year={season}: {before:,} → {len(combined):,} rows "
          f"({len(new):,} fresh {season} rows)")


def main() -> None:
    today = datetime.date.today()
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=today.year,
                    help=f"Season to refresh (default: {today.year})")
    ap.add_argument("--skip-td", action="store_true", dest="skip_td")
    ap.add_argument("--full-rebuild", action="store_true", dest="full_rebuild",
                    help="Refit every end-year instead of splicing current only")
    args = ap.parse_args()
    S = args.season

    print(f"\n### Updating WNBA {S} from wehoop ({today}) ###")
    overall = time.time()

    # 1. pull + merge possessions
    run(f"Pull {S} possessions from wehoop",
        [PY, str(SCRIPTS / "build_season_from_espn.py"), "--season", str(S)])

    # 2. RAPM (splice or full)
    if args.full_rebuild:
        run("Rebuild v6 RAPM (all years)",
            [PY, str(SCRIPTS / "wnba_rapm_v6.py")])
    else:
        run(f"Recompute v6 RAPM for end_year={S}",
            [PY, str(SCRIPTS / "wnba_rapm_v6.py"),
             "--end-year", str(S), "--output", str(TMP_V6)])
        print("\n▶ Splicing into full results file …")
        splice_endyear(S)

    # 3. dashboard
    run("Refresh v6 dashboard",
        [PY, str(SCRIPTS / "format_rapm_dashboard.py"),
         "--input", str(RESULTS_V6), "--output", str(DASH_V6)])

    # 4. TD snapshot
    if not args.skip_td:
        run("Refresh career-decay TD-RAPM snapshot",
            [PY, str(SCRIPTS / "td_rapm" / "wnba_rapm_td.py")])

    # summary
    print(f"\n{'='*70}\n✅ Update complete in {time.time()-overall:.0f}s")
    poss = pd.read_csv(ROOT / "data" / "wnba_possessions_all.csv",
                       usecols=["season", "gameId"])
    cur = poss[poss["season"] == S]
    print(f"   {S}: {cur['gameId'].nunique()} games, {len(cur):,} possessions")
    print(f"   Outputs refreshed:")
    print(f"     • data/wnba_rapm_results_v6.csv")
    print(f"     • data/wnba_rapm_dashboard_v6.csv")
    if not args.skip_td:
        print(f"     • data/td_rapm/wnba_rapm_td.csv")


if __name__ == "__main__":
    main()
