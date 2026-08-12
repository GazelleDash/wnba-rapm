"""
Build player name lookup for WNBA from shufinskiy/nba_data sources.

Combines:
  • wnba_nbastats_* (1997–2025): full player names (PLAYER1_NAME, etc.)
  • wnba_cdnnba_*  (2022–2025): last name + initial format

Output: data/wnba_player_names.csv
  player_id, name, name_last, name_i
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from wnba_pbp_parser import _fetch_list, _download_tar_csv

OUT = Path("data/wnba_player_names.csv")
SEASONS = list(range(2009, 2026))   # 2009–2025


def _add_from_nbastats(df: pd.DataFrame, names: dict[int, str]) -> None:
    for col_id, col_name in [
        ("PLAYER1_ID", "PLAYER1_NAME"),
        ("PLAYER2_ID", "PLAYER2_NAME"),
        ("PLAYER3_ID", "PLAYER3_NAME"),
    ]:
        if col_id not in df.columns or col_name not in df.columns:
            continue
        sub = df[[col_id, col_name]].dropna()
        sub = sub[sub[col_id] != 0]
        for pid, nm in zip(sub[col_id].astype(int), sub[col_name].astype(str)):
            if pid not in names and nm and nm.strip():
                names[pid] = nm.strip()


def main() -> None:
    lookup = _fetch_list()
    full_names: dict[int, str] = {}

    # Pull nbastats for each season + playoffs (more recent overrides if conflict — but we keep first)
    for season in SEASONS:
        for key_tmpl in (f"wnba_nbastats_{season}", f"wnba_nbastats_po_{season}"):
            if key_tmpl not in lookup:
                continue
            try:
                df = _download_tar_csv(key_tmpl, lookup)
                _add_from_nbastats(df, full_names)
                print(f"  {key_tmpl}: {len(full_names)} unique names accumulated")
            except Exception as e:
                print(f"  skip {key_tmpl}: {e}")

    # Build initials format: "Brittney Griner" → "B. Griner"
    rows = []
    for pid, full in full_names.items():
        parts = full.split()
        if len(parts) == 1:
            last  = parts[0]
            ini   = parts[0]
        else:
            last  = parts[-1]
            ini   = f"{parts[0][0]}. {parts[-1]}"
        rows.append({
            "player_id": pid,
            "name":      full,
            "name_last": last,
            "name_i":    ini,
        })

    out = pd.DataFrame(rows).sort_values("player_id").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nSaved {len(out):,} player names → {OUT}")
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
