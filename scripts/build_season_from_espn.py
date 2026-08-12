"""
Build a single WNBA season's possessions from the ESPN/wehoop bulk PBP parquet and merge
into data/wnba_possessions_all.csv — using a verified ESPN→NBA ID crosswalk so
the new season shares player/team IDs with the 2009–2025 NBA-sourced data.

Why this exists
───────────────
shufinskiy/nba_data only publishes completed seasons, and stats.nba.com's
per-game API rate-limits aggressive pulls. sportsdataverse/wehoop publishes the
entire in-progress season as ONE parquet (ESPN-sourced), which downloads in a
single request with no throttling. The only cost is that ESPN uses different
player/team IDs, which this script crosswalks back to NBA IDs.

Pipeline
────────
  1. Download the season parquet (ESPN play-by-play).
  2. Build the ESPN→NBA player crosswalk:
       • exact normalized-name match against data/wnba_player_names.csv
       • 3 verified returning-vet name-change overrides (see VET_OVERRIDES)
       • everyone else = genuinely new player → synthetic ID 90_000_000+espn_id
         (above the NBA player-ID range, below the 10-digit team-ID range)
  3. Build the ESPN→NBA team crosswalk (incl. 2026 expansion TOR/POR).
  4. normalize_espn(): ESPN rows → cdnnba-shaped events with NBA IDs.
  5. Reuse parse_game() (the same parser used for every other season).
  6. Save data/wnba_possessions_2026.csv and merge into possessions_all.csv
     (idempotent: any prior 2026 rows are replaced).
  7. Append any newly-assigned players to data/wnba_player_names.csv.

Usage
─────
  python scripts/build_season_from_espn.py
  python scripts/build_season_from_espn.py --season 2026 --dry-run
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from wnba_pbp_parser import parse_game

WEHOOP_URL = (
    "https://github.com/sportsdataverse/wehoop-wnba-data/raw/main/"
    "wnba/pbp/parquet/play_by_play_{season}.parquet"
)

DEFAULT_ALL   = Path("data/wnba_possessions_all.csv")
DEFAULT_NAMES = Path("data/wnba_player_names.csv")
DEFAULT_TEAMS = Path("data/wnba_team_names.csv")
DEFAULT_DATES = Path("data/wnba_game_dates.csv")

NEW_PLAYER_OFFSET = 90_000_000   # synthetic-ID base for players new to our data

# Verified by hand (returning veterans whose ESPN name differs from our record)
VET_OVERRIDES = {
    "Cheyenne Parker-Tyus": 204323,    # married → "Parker-Tyus"; was "Cheyenne Parker"
    "Skylar Diggins":       203400,    # dropped "-Smith" in 2025
    "Lexi Held":            1631118,   # "Lexi" = "Alexa Held"
}

# ESPN team abbrev → our tricode (where they differ)
ESPN_ABBR_TO_TRICODE = {
    "LA": "LAS", "NY": "NYL", "PHX": "PHO", "WSH": "WAS",
    "LV": "LVA", "GS": "GSV", "CON": "CON", "DAL": "DAL",
    "IND": "IND", "MIN": "MIN", "SEA": "SEA", "CHI": "CHI",
    "ATL": "ATL", "TOR": "TOR", "POR": "POR",
}

# 2026 expansion teams not yet in wnba_team_names.csv → assign next sequential IDs
EXPANSION_TEAMS = {
    "TOR": 1611661332,   # Toronto Tempo
    "POR": 1611661333,   # Portland Fire
}


# ── network ─────────────────────────────────────────────────────────────────────

def fetch_bytes(url: str, attempts: int = 5, timeout: int = 120) -> bytes:
    """GET with retry/backoff.

    GitHub's file CDN intermittently closes the connection mid-transfer
    ("RemoteDisconnected"). A bare requests.get() then kills the whole run,
    including the unattended daily Action. Each attempt uses a fresh session to
    avoid reusing a poisoned keep-alive socket.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            with requests.Session() as s:
                s.headers.update({"User-Agent": "Mozilla/5.0"})
                r = s.get(url, timeout=timeout)
                r.raise_for_status()
                return r.content
        except Exception as e:  # noqa: BLE001 - retry transient network issues
            last = e
            if i < attempts - 1:
                wait = 2 ** i
                print(
                    f"  fetch failed ({type(e).__name__}); "
                    f"retry {i + 1}/{attempts - 1} in {wait}s",
                    flush=True,
                )
                time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url} after {attempts} attempts: {last}")


# ── name normalization / crosswalk ──────────────────────────────────────────────

def norm_name(s: str) -> str:
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[.''`\-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_espn_players(pbp: pd.DataFrame) -> dict[int, str]:
    """ESPN athlete_id → display name, from substitution + shot text."""
    id_to_name: dict[int, str] = {}
    for _, r in pbp[pbp["type_text"] == "Substitution"].iterrows():
        m = re.match(r"(.+?) enters the game for (.+)", str(r["text"]))
        if m:
            if pd.notna(r["athlete_id_1"]):
                id_to_name[int(r["athlete_id_1"])] = m.group(1).strip()
            if pd.notna(r["athlete_id_2"]):
                id_to_name[int(r["athlete_id_2"])] = m.group(2).strip()
    for _, r in pbp[pbp["shooting_play"] == True].iterrows():   # noqa: E712
        if pd.notna(r["athlete_id_1"]):
            m = re.match(r"(.+?) (?:makes|misses)", str(r["text"]))
            if m:
                id_to_name.setdefault(int(r["athlete_id_1"]), m.group(1).strip())
    return id_to_name


def build_player_crosswalk(
    espn_players: dict[int, str],
    names_df: pd.DataFrame,
) -> tuple[dict[int, int], list[dict]]:
    """
    Return (espn_id → nba_id, new_player_rows).
    new_player_rows are dicts ready to append to wnba_player_names.csv.
    """
    nba_map: dict[str, int] = {}
    for _, r in names_df.iterrows():
        nba_map.setdefault(norm_name(r["name"]), int(r["player_id"]))

    vet_norm = {norm_name(k): v for k, v in VET_OVERRIDES.items()}

    xwalk: dict[int, int] = {}
    new_rows: list[dict] = []
    n_exact = n_vet = n_new = 0

    for espn_id, name in sorted(espn_players.items()):
        nn = norm_name(name)
        if nn in vet_norm:
            xwalk[espn_id] = vet_norm[nn]; n_vet += 1
        elif nn in nba_map:
            xwalk[espn_id] = nba_map[nn]; n_exact += 1
        else:
            new_id = NEW_PLAYER_OFFSET + espn_id
            xwalk[espn_id] = new_id
            parts = name.split()
            new_rows.append({
                "player_id": new_id,
                "name":      name,
                "name_last": parts[-1] if parts else name,
                "name_i":    f"{parts[0][0]}. {parts[-1]}" if len(parts) > 1 else name,
            })
            n_new += 1

    print(f"  Crosswalk: {n_exact} exact + {n_vet} vet-override + {n_new} new = "
          f"{len(xwalk)} players")
    return xwalk, new_rows


def build_team_crosswalk(
    pbp: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> tuple[dict[int, int], list[dict]]:
    """ESPN team_id → NBA team_id; plus any expansion-team rows to add."""
    tricode_to_id = {r["tricode"]: int(r["team_id"]) for _, r in teams_df.iterrows()}

    espn_teams = pd.concat([
        pbp[["home_team_id", "home_team_abbrev"]]
            .rename(columns={"home_team_id": "id", "home_team_abbrev": "abbr"}),
        pbp[["away_team_id", "away_team_abbrev"]]
            .rename(columns={"away_team_id": "id", "away_team_abbrev": "abbr"}),
    ]).dropna().drop_duplicates()

    xwalk: dict[int, int] = {}
    new_team_rows: list[dict] = []
    for _, r in espn_teams.iterrows():
        espn_id = int(r["id"])
        tricode = ESPN_ABBR_TO_TRICODE.get(r["abbr"], r["abbr"])
        if tricode in tricode_to_id:
            xwalk[espn_id] = tricode_to_id[tricode]
        elif tricode in EXPANSION_TEAMS:
            nba_id = EXPANSION_TEAMS[tricode]
            xwalk[espn_id] = nba_id
            tricode_to_id[tricode] = nba_id
            new_team_rows.append({"team_id": nba_id, "tricode": tricode})
        else:
            print(f"  ⚠ Unknown ESPN team abbr {r['abbr']} (id {espn_id}) — skipped")
    return xwalk, new_team_rows


# ── ESPN → cdnnba-shaped event normalizer ───────────────────────────────────────

def _ft_subtype(type_text: str) -> str:
    if "Technical" in type_text:
        return "technical"
    m = re.search(r"(\d) of (\d)", type_text)
    return f"{m.group(1)} of {m.group(2)}" if m else "1 of 1"


def normalize_espn(
    pbp: pd.DataFrame,
    player_x: dict[int, int],
    team_x: dict[int, int],
) -> pd.DataFrame:
    """Convert ESPN play-by-play into the cdnnba-shaped schema parse_game wants."""
    pbp = pbp.sort_values(["game_id", "sequence_number"]).reset_index(drop=True)
    out: list[dict] = []
    last_shot_team: dict[int, int] = {}

    for r in pbp.itertuples(index=False):
        gid    = int(r.game_id)
        period = int(r.period_number) if pd.notna(r.period_number) else 0
        tt     = str(r.type_text) if pd.notna(r.type_text) else ""
        text   = str(r.text) if pd.notna(r.text) else ""
        seq    = int(r.sequence_number) if pd.notna(r.sequence_number) else len(out)

        team = team_x.get(int(r.team_id), 0) if pd.notna(r.team_id) else 0
        p1   = player_x.get(int(r.athlete_id_1), 0) if pd.notna(r.athlete_id_1) else 0
        p2   = player_x.get(int(r.athlete_id_2), 0) if pd.notna(r.athlete_id_2) else 0

        tl = text.lower()
        made   = "makes" in tl
        missed = "misses" in tl
        result = "Made" if made else ("Missed" if missed else "")

        action, sub, person, possession = "", "", p1, 0

        # ── Substitution: athlete_1 enters, athlete_2 leaves ──────────────────
        if tt == "Substitution":
            base = {"actionNumber": seq, "period": period, "gameId": gid,
                    "teamId": team, "possession": 0, "shotResult": "",
                    "description": text}
            out.append({**base, "actionType": "substitution",
                        "subType": "in",  "personId": p1})
            out.append({**base, "actionType": "substitution",
                        "subType": "out", "personId": p2})
            continue

        # ── Rebounds (ESPN labels O/D directly) ───────────────────────────────
        elif tt == "Offensive Rebound":
            action, sub, possession = "rebound", "offensive", team
        elif tt == "Defensive Rebound":
            action, sub = "rebound", "defensive"
            possession = last_shot_team.get(gid, team)   # prev offense (shooter)

        # ── Shots ─────────────────────────────────────────────────────────────
        elif r.shooting_play and "Free Throw" not in tt:
            is3 = ("three point" in tl) or (pd.notna(r.score_value) and int(r.score_value) == 3)
            action = "3pt" if is3 else "2pt"
            result = "Made" if made else "Missed"
            possession = team
            last_shot_team[gid] = team

        # ── Free throws ───────────────────────────────────────────────────────
        elif tt.startswith("Free Throw"):
            action = "freethrow"
            sub    = _ft_subtype(tt)
            result = "Made" if made else "Missed"
            possession = team

        # ── Turnovers (incl. "Traveling") ─────────────────────────────────────
        elif "Turnover" in tt or tt == "Traveling":
            action, possession = "turnover", team

        # ── Fouls ─────────────────────────────────────────────────────────────
        elif "Foul" in tt:
            action = "foul"
            sub    = "technical" if "Technical" in tt else "personal"

        # ── Period / game end ─────────────────────────────────────────────────
        elif tt in ("End Period", "End Game"):
            action, sub = "period", "end"

        # ── Jump ball ─────────────────────────────────────────────────────────
        elif tt in ("Jumpball", "Jump Ball"):
            action, sub = "jumpball", "recovered"

        else:
            continue   # timeouts, reviews, challenges, violations — skip

        out.append({
            "actionNumber": seq, "period": period, "gameId": gid,
            "actionType": action, "subType": sub,
            "personId": person, "teamId": team,
            "possession": possession, "shotResult": result,
            "description": text,
        })

    return pd.DataFrame(out)


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season",  type=int, default=2026)
    ap.add_argument("--all-csv", type=Path, default=DEFAULT_ALL, dest="all_csv")
    ap.add_argument("--names",   type=Path, default=DEFAULT_NAMES)
    ap.add_argument("--teams",   type=Path, default=DEFAULT_TEAMS)
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    args = ap.parse_args()

    print(f"Downloading ESPN {args.season} PBP …")
    url = WEHOOP_URL.format(season=args.season)
    pbp = pd.read_parquet(io.BytesIO(fetch_bytes(url)))
    print(f"  {len(pbp):,} plays, {pbp['game_id'].nunique()} games")

    names_df = pd.read_csv(args.names)
    teams_df = pd.read_csv(args.teams)

    # ── crosswalks ────────────────────────────────────────────────────────────
    print("Building crosswalks …")
    espn_players = extract_espn_players(pbp)
    player_x, new_player_rows = build_player_crosswalk(espn_players, names_df)
    team_x, new_team_rows     = build_team_crosswalk(pbp, teams_df)
    for t in new_team_rows:
        print(f"  + expansion team {t['tricode']} = {t['team_id']}")

    # ── normalize + parse per game ────────────────────────────────────────────
    print("Normalizing + parsing games …")
    norm = normalize_espn(pbp, player_x, team_x)
    all_rows: list[dict] = []
    game_ids = norm["gameId"].unique()
    for i, gid in enumerate(game_ids, 1):
        gdf = norm[norm["gameId"] == gid]
        all_rows.extend(parse_game(gdf))
        print(f"  [{i:2d}/{len(game_ids)}] game {gid}", end="\r")
    print()

    result = pd.DataFrame(all_rows)
    result["season"]     = args.season
    result["seasonType"] = "regular"
    complete = int(result["lineup_complete"].sum())
    print(f"  Parsed {len(result):,} possessions "
          f"({complete:,} complete, {complete/len(result)*100:.1f}%)")

    if args.dry_run:
        print("\n[dry-run] Top scorers' possession counts (sanity check):")
        _sanity(result, names_df, new_player_rows)
        return

    # ── write season file ─────────────────────────────────────────────────────
    season_out = Path(f"data/wnba_possessions_{args.season}.csv")
    result.to_csv(season_out, index=False)
    print(f"  Saved → {season_out}")

    # ── record game dates so time decay works on this season ─────────────────
    # The archived date table only covers NBA-sourced game IDs, so without this
    # every ESPN game would be missing a date and fall back to full weight
    # (i.e. no decay) inside the RAPM fit.
    game_dates = (pbp[["game_id", "game_date"]].dropna()
                    .drop_duplicates("game_id")
                    .rename(columns={"game_id": "game_id", "game_date": "game_date"}))
    game_dates["game_date"] = pd.to_datetime(game_dates["game_date"]).dt.strftime("%Y-%m-%d")
    if DEFAULT_DATES.exists():
        existing_dates = pd.read_csv(DEFAULT_DATES)
        game_dates = pd.concat([existing_dates, game_dates], ignore_index=True)
    game_dates = game_dates.drop_duplicates("game_id", keep="last")
    game_dates.to_csv(DEFAULT_DATES, index=False)
    print(f"  Game dates → {DEFAULT_DATES} ({len(game_dates):,} total)")

    # ── append new players / teams ────────────────────────────────────────────
    if new_player_rows:
        updated = pd.concat([names_df, pd.DataFrame(new_player_rows)], ignore_index=True)
        updated = updated.drop_duplicates("player_id").sort_values("player_id")
        updated.to_csv(args.names, index=False)
        print(f"  +{len(new_player_rows)} new players → {args.names}")
    if new_team_rows:
        updated_t = pd.concat([teams_df, pd.DataFrame(new_team_rows)], ignore_index=True)
        updated_t = updated_t.drop_duplicates("team_id")
        updated_t.to_csv(args.teams, index=False)
        print(f"  +{len(new_team_rows)} new teams → {args.teams}")

    # ── merge into possessions_all (idempotent) ───────────────────────────────
    if args.all_csv.exists():
        existing = pd.read_csv(args.all_csv)
        before = len(existing)
        existing = existing[~((existing["season"] == args.season) &
                              (existing["seasonType"] == "regular"))]
        if before - len(existing):
            print(f"  Replaced {before - len(existing):,} existing {args.season} rows")
        combined = pd.concat([existing, result], ignore_index=True)
    else:
        combined = result
    combined = combined.sort_values(["season", "gameId", "period"]).reset_index(drop=True)
    combined.to_csv(args.all_csv, index=False)
    print(f"  Updated {args.all_csv}: {len(combined):,} rows, "
          f"{combined['season'].nunique()} seasons "
          f"({combined['season'].min()}–{combined['season'].max()})")


def _sanity(result, names_df, new_player_rows):
    """Print possession leaders so we can eyeball the crosswalk before merging."""
    name_map = {int(r["player_id"]): r["name"] for _, r in names_df.iterrows()}
    for r in new_player_rows:
        name_map[r["player_id"]] = r["name"]
    from collections import Counter
    cnt = Counter()
    for _, row in result[result["lineup_complete"] == 1].iterrows():
        for c in [f"offensePlayer{i}Id" for i in range(1, 6)]:
            cnt[int(row[c])] += 1
    for pid, n in cnt.most_common(15):
        print(f"    {name_map.get(pid, '??? id='+str(pid)):24s} {n:4d} off poss")


if __name__ == "__main__":
    main()
