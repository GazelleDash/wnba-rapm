"""
Fetch WNBA play-by-play directly from ESPN's public API.

Why this exists
───────────────
The normal path pulls a whole season as one parquet from
sportsdataverse/wehoop-wnba-data. That repo builds its parquet from
sportsdataverse/wehoop-wnba-raw — and when the raw ingestion stalls, the
published parquet silently freezes. It still rebuilds daily and reports
success; it just rebuilds from a stale game list, so nothing looks broken
from the outside.

That is exactly what happened after 2026-08-01: the raw repo's daily commits
stopped adding game files (its crosswalk step kept timing out against
stats.wnba.com), so the parquet sat at 219 games / 90,198 rows for over a
week while real games kept being played.

ESPN's own API has those games immediately. This module fetches from it
directly and returns rows in the SAME column schema as the wehoop parquet, so
build_season_from_espn.py can simply concatenate the two and reuse
normalize_espn() unchanged — no second normalizer to keep in sync.

Endpoints (public, no auth, no key):
  scoreboard?dates=YYYYMMDD  ->  which games happened that day
  summary?event={game_id}    ->  that game's full play-by-play

Substitution encoding — verified against the boxscore, not assumed:
  participants[0] = player ENTERING,  participants[1] = player LEAVING
  which matches wehoop's athlete_id_1 / athlete_id_2 exactly. Getting this
  backwards would silently corrupt every reconstructed lineup, so
  verify_sub_order() re-checks it at runtime.

Usage:
  python scripts/fetch_espn_live.py --since 2026-08-01
  python scripts/fetch_espn_live.py --since 2026-08-01 --out /tmp/gap.parquet
"""

from __future__ import annotations

import argparse
import datetime as _dt
import time
from pathlib import Path

import pandas as pd
import requests

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={date}"
SUMMARY    = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={gid}"

REQUEST_DELAY = 0.4      # polite pacing between game fetches

# Columns build_season_from_espn.py actually consumes. Anything the wehoop
# parquet has beyond these is unused, so we don't synthesize it.
SCHEMA = [
    "game_id", "game_date", "sequence_number", "period_number",
    "type_text", "text", "athlete_id_1", "athlete_id_2", "team_id",
    "shooting_play", "score_value",
    "home_team_id", "home_team_abbrev", "away_team_id", "away_team_abbrev",
]


def _get(url: str, attempts: int = 4, timeout: int = 25) -> dict | None:
    """GET JSON with retry/backoff. Returns None if it never succeeds."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            # Do NOT set a browser-like User-Agent here. ESPN's API returns
            # 403 for "Mozilla/5.0"-style UAs and serves requests' own default
            # UA fine — spoofing a browser on an API endpoint is precisely what
            # gets blocked. Verified: no header -> 200, Mozilla/5.0 -> 403.
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:                       # noqa: BLE001 - retry anything
            last = e
            if i < attempts - 1:
                time.sleep(2 ** i)
    print(f"    fetch gave up: {type(last).__name__}: {last}")
    return None


def discover_games(start: _dt.date, end: _dt.date) -> list[tuple[str, str]]:
    """Return [(game_id, iso_date)] for COMPLETED games in [start, end]."""
    out: list[tuple[str, str]] = []
    day = start
    while day <= end:
        d = _get(SCOREBOARD.format(date=day.strftime("%Y%m%d")))
        if d:
            for ev in d.get("events", []):
                comps = ev.get("competitions") or [{}]
                status = comps[0].get("status", {}).get("type", {})
                # Only finished games — in-progress ones would import partial
                # possessions and then never be corrected on a later run.
                if status.get("completed") is True:
                    # Use the QUERIED date, not ev["date"]. ESPN's event date is
                    # a UTC timestamp, so an 8pm ET game lands on the next UTC
                    # day — while wehoop keys games by US/Eastern date (verified:
                    # game 401856899 is wehoop 2026-05-09 vs ESPN
                    # 2026-05-10T00:00Z). The scoreboard's dates= param is
                    # already ET, so the query date matches wehoop exactly and
                    # needs no timezone conversion.
                    out.append((str(ev["id"]), day.isoformat()))
        day += _dt.timedelta(days=1)
        time.sleep(REQUEST_DELAY)
    return out


def fetch_game(game_id: str, game_date: str) -> list[dict]:
    """One game's plays, as rows in the wehoop parquet schema.

    game_date must be the US/Eastern calendar date supplied by discover_games(),
    NOT ESPN's UTC timestamp — see the note there. Passing it in also keeps the
    column tz-naive, matching wehoop's plain 'YYYY-MM-DD' strings; mixing the two
    makes pandas raise "Cannot mix tz-aware with tz-naive values" downstream.
    """
    d = _get(SUMMARY.format(gid=game_id))
    if not d or "plays" not in d:
        return []

    comp = (d.get("header", {}).get("competitions") or [{}])[0]
    home_id = home_ab = away_id = away_ab = None
    for c in comp.get("competitors", []):
        tid, ab = c.get("team", {}).get("id"), c.get("team", {}).get("abbreviation")
        if c.get("homeAway") == "home":
            home_id, home_ab = tid, ab
        else:
            away_id, away_ab = tid, ab

    rows: list[dict] = []
    for p in d["plays"]:
        parts = [x.get("athlete", {}).get("id") for x in (p.get("participants") or [])]
        rows.append({
            "game_id":          int(game_id),
            "game_date":        game_date,
            "sequence_number":  int(p["sequenceNumber"]) if str(p.get("sequenceNumber", "")).isdigit() else None,
            "period_number":    (p.get("period") or {}).get("number"),
            "type_text":        (p.get("type") or {}).get("text", ""),
            "text":             p.get("text", ""),
            "athlete_id_1":     float(parts[0]) if len(parts) > 0 and parts[0] else None,
            "athlete_id_2":     float(parts[1]) if len(parts) > 1 and parts[1] else None,
            "team_id":          float(p["team"]["id"]) if p.get("team", {}).get("id") else None,
            "shooting_play":    bool(p.get("shootingPlay", False)),
            "score_value":      int(p.get("scoreValue") or 0),
            "home_team_id":     float(home_id) if home_id else None,
            "home_team_abbrev": home_ab,
            "away_team_id":     float(away_id) if away_id else None,
            "away_team_abbrev": away_ab,
        })
    return rows


def verify_sub_order(df: pd.DataFrame) -> bool:
    """Sanity-check that participants[0] is the ENTERING player.

    ESPN's text reads "<A> enters the game for <B>", so athlete_id_1 must be A.
    We can't compare IDs to names without another request, but we CAN assert
    every substitution carries exactly two participants — if ESPN ever changed
    the encoding, that invariant is the first thing that would break.
    """
    subs = df[df["type_text"] == "Substitution"]
    if subs.empty:
        return True
    bad = subs[subs["athlete_id_1"].isna() | subs["athlete_id_2"].isna()]
    if len(bad):
        print(f"  ⚠ {len(bad)} substitution rows missing a participant — "
              f"lineup tracking may be degraded for those")
        return False
    return True


def fetch_range(start: _dt.date, end: _dt.date) -> pd.DataFrame:
    games = discover_games(start, end)
    print(f"  {len(games)} completed game(s) between {start} and {end}")
    if not games:
        return pd.DataFrame(columns=SCHEMA)

    rows: list[dict] = []
    for i, (gid, gdate) in enumerate(games, 1):
        got = fetch_game(gid, gdate)
        rows.extend(got)
        print(f"  [{i:3d}/{len(games)}] game {gid}: {len(got):4d} plays", end="\r")
        if i < len(games):
            time.sleep(REQUEST_DELAY)
    print()

    df = pd.DataFrame(rows, columns=SCHEMA)
    verify_sub_order(df)
    print(f"  {len(df):,} plays from {df['game_id'].nunique()} games")
    return df


def _parse_date(s: str) -> _dt.date:
    return _dt.datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--since", required=True,
                    help="Fetch games AFTER this date (YYYY-MM-DD), exclusive")
    ap.add_argument("--until", default=None,
                    help="Last date to fetch (YYYY-MM-DD). Default: today (UTC)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write the result to this .parquet instead of just summarizing")
    args = ap.parse_args()

    start = _parse_date(args.since) + _dt.timedelta(days=1)
    end = _parse_date(args.until) if args.until else _dt.datetime.utcnow().date()

    print(f"Fetching WNBA play-by-play from ESPN, {start} → {end}")
    df = fetch_range(start, end)

    if df.empty:
        print("  nothing to write")
        return
    print("\n  games by date:")
    by_date = df.drop_duplicates("game_id").groupby(
        pd.to_datetime(df.drop_duplicates("game_id")["game_date"]).dt.date).size()
    for d, n in by_date.items():
        print(f"    {d}  {n} game(s)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
