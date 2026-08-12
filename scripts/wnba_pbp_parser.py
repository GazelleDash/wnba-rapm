"""
scripts/wnba_pbp_parser.py

Parse WNBA play-by-play (via shufinskiy/nba_data) into one-way possession
rows ready for RAPM calculation.

Supports two source formats from the same repo:
    cdnnba   (cdn.nba.com)   — 2022–2025  (preferred: richer fields)
    datanba  (data.nba.com)  — 2017–2025  (fallback for older seasons)

Auto-picks cdnnba when available, else falls back to datanba.

Output columns per possession:
    offensePlayer1Id..5  — offensive player IDs (0 = unknown / incomplete)
    defensePlayer1Id..5  — defensive player IDs
    points               — points scored by the offense
    possessions          — always 1
    fga, fga3            — field-goal attempts (total, 3-point)
    ftm, fta             — free throws made / attempted by offense
    off_reb              — offensive rebounds
    def_reb              — defensive rebounds (captured by defense)
    turnovers            — turnovers committed by offense
    gameId, period, offenseTeamId, defenseTeamId
    lineup_complete      — 1 if all 10 player IDs are known, else 0

Usage:
    python scripts/wnba_pbp_parser.py --season 2024
    python scripts/wnba_pbp_parser.py --season 2017 2018 2019 2020 2021 2022 2023 2024
    python scripts/wnba_pbp_parser.py --all                      # 2017→2025
    python scripts/wnba_pbp_parser.py --season 2024 --playoffs
"""

from __future__ import annotations

import argparse
import io
import tarfile
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

LIST_DATA_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt"

_FT_LAST = frozenset({"1 of 1", "2 of 2", "3 of 3"})
_SKIP_ACTIONS = frozenset({"game"})          # never added to possessions
_LINEUP_SKIP = frozenset({                   # don't infer starter from these
    "period", "timeout", "game", "jumpball", "substitution",
})


# ── Data loading ──────────────────────────────────────────────────────────────

def _fetch_list() -> dict[str, str]:
    with urlopen(LIST_DATA_URL) as f:
        lines = f.read().decode("utf-8").strip().split("\n")
    return {ln.split("=")[0]: ln.split("=")[1] for ln in lines if "=" in ln}


def _download_tar_csv(key: str, lookup: dict[str, str]) -> pd.DataFrame:
    if key not in lookup:
        return pd.DataFrame()
    url = lookup[key]
    print(f"  Downloading {key} ...")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r:
        content = r.read()
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:xz") as tar:
        csv_file = tar.extractfile(f"{key}.csv")
        return pd.read_csv(csv_file)


def load_wnba_cdnnba(season: int, playoffs: bool = False) -> pd.DataFrame:
    """Download and return cdnnba PBP for one WNBA season."""
    lookup = _fetch_list()
    key = f"wnba_cdnnba_po_{season}" if playoffs else f"wnba_cdnnba_{season}"
    df = _download_tar_csv(key, lookup)
    if df.empty:
        avail = sorted(k for k in lookup if k.startswith("wnba_cdnnba"))
        raise ValueError(f"{key!r} not found.\nAvailable: {avail}")
    print(f"  Loaded {len(df):,} events across {df['gameId'].nunique()} games")
    return df


def load_wnba_datanba(season: int, playoffs: bool = False) -> pd.DataFrame:
    """Download and return datanba PBP for one WNBA season (older format)."""
    lookup = _fetch_list()
    key = f"wnba_datanba_po_{season}" if playoffs else f"wnba_datanba_{season}"
    df = _download_tar_csv(key, lookup)
    if df.empty:
        avail = sorted(k for k in lookup if k.startswith("wnba_datanba"))
        raise ValueError(f"{key!r} not found.\nAvailable: {avail}")
    print(f"  Loaded {len(df):,} events across {df['GAME_ID'].nunique()} games")
    return df


def load_wnba_nbastats(season: int, playoffs: bool = False) -> pd.DataFrame:
    """Download and return nbastats PBP for one WNBA season (covers 1997+)."""
    lookup = _fetch_list()
    key = f"wnba_nbastats_po_{season}" if playoffs else f"wnba_nbastats_{season}"
    df = _download_tar_csv(key, lookup)
    if df.empty:
        avail = sorted(k for k in lookup if k.startswith("wnba_nbastats"))
        raise ValueError(f"{key!r} not found.\nAvailable: {avail}")
    print(f"  Loaded {len(df):,} events across {df['GAME_ID'].nunique()} games")
    return df


def load_wnba_pbp(season: int, playoffs: bool = False) -> tuple[pd.DataFrame, str]:
    """
    Load PBP for a season, auto-picking the best available format.
    Returns (dataframe, format_name) where format_name is one of:
      'cdnnba'   (2022+ — richest)
      'datanba'  (2017–2025 — fallback)
      'nbastats' (1997+ — only path for pre-2017 seasons)
    """
    lookup = _fetch_list()
    cdn_key = f"wnba_cdnnba_po_{season}" if playoffs else f"wnba_cdnnba_{season}"
    if cdn_key in lookup:
        return load_wnba_cdnnba(season, playoffs), "cdnnba"
    dn_key = f"wnba_datanba_po_{season}" if playoffs else f"wnba_datanba_{season}"
    if dn_key in lookup:
        return load_wnba_datanba(season, playoffs), "datanba"
    return load_wnba_nbastats(season, playoffs), "nbastats"


# ── datanba → cdnnba normalization ────────────────────────────────────────────

# datanba mtype → cdnnba freethrow subType
_FT_MTYPE_TO_SUB = {
    10: "1 of 1", 11: "1 of 2", 12: "2 of 2",
    13: "1 of 3", 14: "2 of 3", 15: "3 of 3",
    16: "technical",
}

# datanba mtype → cdnnba foul subType (best-effort)
_FOUL_MTYPE_TO_SUB = {
    1: "personal", 2: "personal", 3: "personal", 4: "offensive",
    5: "personal", 6: "personal", 9: "personal",
    11: "technical", 13: "technical", 16: "technical",
    14: "personal", 15: "personal", 26: "offensive",
}


def _datanba_action(etype: int, mtype: int, opt1: int, de: str, tid: int, oftid: int) -> tuple[str, str, str]:
    """Map a datanba (etype, mtype, ...) tuple → (actionType, subType, shotResult)."""
    de_lower = de.lower()

    if etype == 1:  # made FG
        return ("3pt" if opt1 == 3 else "2pt"), "", "Made"
    if etype == 2:  # missed FG
        return ("3pt" if opt1 == 3 else "2pt"), "", "Missed"
    if etype == 3:  # FT
        sub = _FT_MTYPE_TO_SUB.get(mtype, "")
        result = "Missed" if "missed" in de_lower else "Made"
        return "freethrow", sub, result
    if etype == 4:  # rebound — derive off/def from team-vs-offense team
        return "rebound", ("offensive" if tid == oftid else "defensive"), ""
    if etype == 5:
        return "turnover", "", ""
    if etype == 6:
        return "foul", _FOUL_MTYPE_TO_SUB.get(mtype, "personal"), ""
    if etype == 7:
        return "violation", "", ""
    if etype == 9:
        return "timeout", "", ""
    if etype == 10:
        return "jumpball", "recovered", ""
    if etype == 11:
        return "ejection", "", ""
    if etype == 12:
        return "period", "start", ""
    if etype == 13:
        return "period", "end", ""
    return "", "", ""


def normalize_datanba(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a datanba PBP DataFrame into cdnnba-shaped records."""
    out: list[dict] = []
    for r in df.sort_values(["GAME_ID", "evt"]).itertuples(index=False):
        etype = _sint(r.etype)
        mtype = _sint(r.mtype)
        opt1  = _sint(r.opt1)
        tid   = _sint(r.tid)
        oftid = _sint(r.oftid)
        de    = str(r.de) if r.de == r.de else ""
        evt   = _sint(r.evt)
        period = _sint(r.PERIOD)
        gid    = _sint(r.GAME_ID)

        # Substitutions: expand into two cdnnba-style rows (out + in)
        if etype == 8:
            pid_out = _sint(r.pid)
            pid_in  = _sint(r.epid)
            base = {
                "actionNumber": evt, "period": period, "gameId": gid,
                "teamId": tid, "possession": oftid, "shotResult": "",
                "description": de,
            }
            out.append({**base, "actionType": "substitution",
                        "subType": "out", "personId": pid_out})
            out.append({**base, "actionType": "substitution",
                        "subType": "in",  "personId": pid_in})
            continue

        action, sub, result = _datanba_action(etype, mtype, opt1, de, tid, oftid)
        out.append({
            "actionNumber": evt, "period": period, "gameId": gid,
            "actionType": action, "subType": sub,
            "personId": _sint(r.pid), "teamId": tid,
            "possession": oftid, "shotResult": result,
            "description": de,
        })

    return pd.DataFrame(out)


# ── nbastats → cdnnba normalization (for 1997–2016 seasons) ───────────────────

# nbastats EVENTMSGACTIONTYPE → cdnnba foul subType (best-effort)
_NBASTATS_FOUL_TO_SUB = {
    1: "personal", 2: "personal", 3: "personal", 4: "offensive",
    5: "personal", 6: "personal", 9: "personal",
    11: "technical", 13: "technical", 16: "technical",
    14: "personal", 15: "personal", 26: "offensive",
}

# nbastats FT subType derived from description "Free Throw N of M"
_NBASTATS_FT_PATTERNS = [
    ("1 of 1", "1 of 1"),
    ("1 of 2", "1 of 2"),
    ("2 of 2", "2 of 2"),
    ("1 of 3", "1 of 3"),
    ("2 of 3", "2 of 3"),
    ("3 of 3", "3 of 3"),
]


def _nbastats_ft_sub(desc: str) -> str:
    """Extract FT sequence ("1 of 2", etc.) from description string."""
    for needle, sub in _NBASTATS_FT_PATTERNS:
        if needle in desc:
            # Tag technical FTs as their own subtype
            if "echnical" in desc:
                return "technical"
            return sub
    return ""


def normalize_nbastats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert an nbastats PBP DataFrame into cdnnba-shaped records.

    Key differences from datanba:
      • EVENTMSGTYPE codes (1=made, 2=missed, 3=FT, 4=reb, 5=TO, 6=foul,
        7=violation, 8=sub, 9=timeout, 10=jump, 11=ejection, 12/13=period)
      • PLAYER1_ID/PLAYER2_ID are out/in for substitutions
      • Description text identifies 3PT shots, FT sequence, technical FTs
      • PLAYER1_ID can be a TEAM ID for team rebounds (10-digit values)
      • No explicit `oftid` → derived by tracking last shooter team
    """
    df = df.sort_values(["GAME_ID", "EVENTNUM"]).reset_index(drop=True)
    out: list[dict] = []
    last_shot_team: dict[int, int] = {}   # game_id → most recent shooting team

    for r in df.itertuples(index=False):
        etype  = _sint(r.EVENTMSGTYPE)
        atype  = _sint(r.EVENTMSGACTIONTYPE)
        gid    = _sint(r.GAME_ID)
        period = _sint(r.PERIOD)
        evt    = _sint(r.EVENTNUM)

        p1_id   = _sint(r.PLAYER1_ID)
        p1_team = _sint(r.PLAYER1_TEAM_ID)
        p2_id   = _sint(r.PLAYER2_ID)
        p2_team = _sint(getattr(r, "PLAYER2_TEAM_ID", 0))

        # Combined description (one of HOMEDESCRIPTION / VISITORDESCRIPTION is set)
        h = str(r.HOMEDESCRIPTION) if r.HOMEDESCRIPTION == r.HOMEDESCRIPTION else ""
        v = str(r.VISITORDESCRIPTION) if r.VISITORDESCRIPTION == r.VISITORDESCRIPTION else ""
        desc = h if h else v

        # Team rebound detection: PLAYER1_ID is a 10-digit team ID, no team field set
        is_team_event = p1_id > 1_000_000_000
        team_id  = p1_team if p1_team else (p1_id if is_team_event else 0)
        person_id = 0 if is_team_event else p1_id

        # Default fields
        action  = ""
        sub     = ""
        result  = ""
        offense_team = 0

        if etype == 1:    # Made shot
            action = "3pt" if "3PT" in desc else "2pt"
            result = "Made"
            offense_team = team_id
            last_shot_team[gid] = team_id
        elif etype == 2:  # Missed shot
            action = "3pt" if "3PT" in desc else "2pt"
            result = "Missed"
            offense_team = team_id
            last_shot_team[gid] = team_id
        elif etype == 3:  # Free throw
            action = "freethrow"
            sub    = _nbastats_ft_sub(desc)
            result = "Missed" if "MISS" in desc else "Made"
            offense_team = team_id
        elif etype == 4:  # Rebound
            action = "rebound"
            shooter = last_shot_team.get(gid, 0)
            if shooter and team_id:
                sub = "offensive" if team_id == shooter else "defensive"
                offense_team = team_id if sub == "offensive" else shooter
            else:
                sub = "defensive"
                offense_team = team_id
        elif etype == 5:  # Turnover
            action = "turnover"
            offense_team = team_id
        elif etype == 6:  # Foul
            action = "foul"
            sub    = _NBASTATS_FOUL_TO_SUB.get(atype, "personal")
            # Technical fouls: detect via description as a backup
            if "Technical" in desc or "T.FOUL" in desc:
                sub = "technical"
        elif etype == 7:  # Violation
            action = "violation"
        elif etype == 8:  # Substitution — emit out + in pair
            base = {
                "actionNumber": evt, "period": period, "gameId": gid,
                "teamId": p1_team, "possession": 0, "shotResult": "",
                "description": desc,
            }
            out.append({**base, "actionType": "substitution",
                        "subType": "out", "personId": p1_id})
            out.append({**base, "actionType": "substitution",
                        "subType": "in",  "personId": p2_id,
                        "teamId":   p2_team or p1_team})
            continue
        elif etype == 9:   # Timeout
            action = "timeout"
        elif etype == 10:  # Jump ball
            action = "jumpball"; sub = "recovered"
        elif etype == 11:  # Ejection
            action = "ejection"
        elif etype == 12:  # Period start
            action = "period"; sub = "start"
        elif etype == 13:  # Period end
            action = "period"; sub = "end"
        else:
            continue       # skip unknown event types (e.g. 18 = stoppage/sponsor)

        out.append({
            "actionNumber": evt, "period": period, "gameId": gid,
            "actionType": action, "subType": sub,
            "personId": person_id, "teamId": team_id,
            "possession": offense_team, "shotResult": result,
            "description": desc,
        })

    return pd.DataFrame(out)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sint(val) -> int:
    """Safe cast to int; NaN / None → 0."""
    try:
        if val != val:   # NaN
            return 0
        return int(val)
    except (TypeError, ValueError):
        return 0


def _is_and1(i: int, ev: list[dict]) -> bool:
    """
    True if ev[i] is a made field goal that is the start of an And-1.
    We look ahead for:  foul (personal)  →  freethrow 1 of 1  by the same player.
    Timeouts and substitutions between those events are skipped.
    """
    shooter = _sint(ev[i].get("personId"))
    found_foul = False
    for j in range(i + 1, min(i + 10, len(ev))):
        a = ev[j].get("actionType", "")
        s = ev[j].get("subType", "") or ""
        if a in ("timeout", "substitution", "ejection"):
            continue
        if a == "foul" and s == "personal" and not found_foul:
            found_foul = True
            continue
        if (a == "freethrow" and s == "1 of 1"
                and _sint(ev[j].get("personId")) == shooter
                and found_foul):
            return True
        # Any other game event breaks the And-1 window
        if a in ("2pt", "3pt", "rebound", "turnover"):
            break
    return False


def _is_technical_ft(i: int, ev: list[dict]) -> bool:
    """
    True if ev[i] (a freethrow) is preceded by a technical foul within
    a short lookback window.
    """
    for j in range(i - 1, max(-1, i - 7), -1):
        a = ev[j].get("actionType", "")
        s = ev[j].get("subType", "") or ""
        if a in ("timeout", "substitution", "ejection"):
            continue
        if a == "foul" and s == "technical":
            return True
        # A possession-relevant event before the tech → stop looking
        if a in ("2pt", "3pt", "rebound", "turnover", "freethrow"):
            break
    return False


def _ends_possession(i: int, ev: list[dict]) -> bool:
    """Return True if the event at index i ends the current possession."""
    row = ev[i]
    a = row.get("actionType", "")
    s = row.get("subType", "") or ""
    r = row.get("shotResult", "") or ""

    if a == "turnover":
        return True
    if a == "rebound" and s == "defensive":
        return True
    if a in ("2pt", "3pt") and r == "Made":
        return not _is_and1(i, ev)
    if a == "freethrow" and s in _FT_LAST and r == "Made":
        # Technical FTs don't end the possession of the team on offense
        return not _is_technical_ft(i, ev)
    if a == "period" and s == "end":
        return True
    return False


# ── Lineup tracking ───────────────────────────────────────────────────────────

def _infer_period_starters(period_events: list[dict], team: int) -> list[int]:
    """
    Return the (up to 5) player IDs that started a period for `team`.

    Uses the FIRST event per player to classify them:
      - first event = sub-out  → was on floor at period start → starter
      - first event = action   → was on floor at period start → starter
      - first event = sub-in   → arrived from bench mid-period → not a starter

    This correctly handles the common case where a player enters mid-period,
    plays, then gets subbed out again (they would look like a starter under
    a naive net-sub-count approach).
    """
    starters: list[int] = []
    seen: set[int] = set()

    for e in period_events:
        t = _sint(e.get("teamId"))
        p = _sint(e.get("personId"))
        a = e.get("actionType", "")
        s = e.get("subType", "") or ""

        if t != team or p <= 0 or p == team:
            continue
        if a == "foul" and s == "technical":   # team-tech placeholder
            continue
        if p in seen:
            continue  # already classified this player

        seen.add(p)

        if a == "substitution":
            if s == "out":
                starters.append(p)   # on floor → starter
            # s == "in" → bench arrival, skip
        elif a not in _LINEUP_SKIP:
            starters.append(p)       # appeared in action → starter

    return starters[:5]


def _build_game_lineups(events: list[dict]) -> list[dict]:
    """
    Two-pass lineup reconstruction.

    Pass 1 (backward): for every (period, team) pair, reconstruct the five
    players who started that period using _infer_period_starters.

    Pass 2 (forward): replay events in order, seeding each period with the
    reconstructed starters, then applying substitutions live.

    Falls back to first-appearance tracking for any player slot still unknown
    after reconstruction (e.g. extremely sparse periods).
    """
    teams = sorted({_sint(e.get("teamId")) for e in events if _sint(e.get("teamId")) > 0})
    if len(teams) < 2:
        seen: list[int] = []
        for e in events:
            t = _sint(e.get("teamId"))
            if t > 0 and t not in seen:
                seen.append(t)
            if len(seen) == 2:
                break
        teams = sorted(seen)
    if len(teams) < 2:
        return events  # degenerate game — return as-is

    t1, t2 = teams[0], teams[1]

    # ── Pass 1: reconstruct starters per (period, team) ──────────────────────
    by_period: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        per = _sint(e.get("period"))
        if per > 0:
            by_period[per].append(e)

    period_starters: dict[tuple[int, int], list[int]] = {}
    for per, pevents in sorted(by_period.items()):
        for team in (t1, t2):
            period_starters[(per, team)] = _infer_period_starters(pevents, team)

    # ── Pass 2: forward tracking ──────────────────────────────────────────────
    lineup: dict[int, list[int]] = {t1: [], t2: []}
    current_period = 0

    result: list[dict] = []
    for e in events:
        per    = _sint(e.get("period"))
        team   = _sint(e.get("teamId"))
        person = _sint(e.get("personId"))
        action = e.get("actionType", "")
        sub    = e.get("subType", "") or ""

        # New period → seed lineup from reconstructed starters
        if per > 0 and per != current_period:
            current_period = per
            for t in (t1, t2):
                lineup[t] = list(period_starters.get((per, t), []))

        if action == "substitution" and team in lineup:
            if sub == "out" and person:
                if person not in lineup[team]:
                    lineup[team].append(person)  # fallback: unseen starter
                lineup[team].remove(person)
            elif sub == "in" and person:
                if person not in lineup[team]:
                    lineup[team].append(person)

        elif action == "ejection" and team in lineup and person:
            if person not in lineup[team]:
                lineup[team].append(person)
            lineup[team].remove(person)

        elif action not in _LINEUP_SKIP and person > 0 and team in lineup:
            is_team_tech = action == "foul" and sub == "technical"
            if person != team and not is_team_tech:
                if person not in lineup[team] and len(lineup[team]) < 5:
                    lineup[team].append(person)  # fallback: first-appearance

        l1 = (lineup[t1] + [0] * 5)[:5]
        l2 = (lineup[t2] + [0] * 5)[:5]
        row = dict(e)
        row["team1_id"] = t1
        row["team2_id"] = t2
        for k in range(5):
            row[f"team1_p{k+1}"] = l1[k]
            row[f"team2_p{k+1}"] = l2[k]
        result.append(row)

    return result


# ── Possession aggregation ────────────────────────────────────────────────────

def _get_offense_team(poss: list[dict]) -> int:
    """
    Return the offensive team ID for this possession.

    If the possession ends with a defensive rebound, the `possession` column
    on that event shows the NEW offense — skip it and look at earlier events.
    """
    last = poss[-1] if poss else {}
    skip_last = (
        last.get("actionType") == "rebound"
        and (last.get("subType") or "") == "defensive"
    )
    seq = poss[:-1] if skip_last else poss
    for e in seq:
        p = _sint(e.get("possession"))
        if p > 0:
            return p
    return 0


def _get_lineups(
    poss: list[dict], offense_team: int, t1: int, t2: int
) -> tuple[list[int], list[int]]:
    """Return (off_lineup, def_lineup) from the first event with lineup data."""
    for e in poss:
        l1 = [_sint(e.get(f"team1_p{k+1}")) for k in range(5)]
        l2 = [_sint(e.get(f"team2_p{k+1}")) for k in range(5)]
        if any(l1) or any(l2):
            return (l1, l2) if offense_team == t1 else (l2, l1)
    return [0] * 5, [0] * 5


def _build_possession_row(poss: list[dict], game_id: int) -> dict | None:
    if not poss:
        return None

    offense_team = _get_offense_team(poss)
    if offense_team == 0:
        return None

    t1 = _sint(poss[0].get("team1_id"))
    t2 = _sint(poss[0].get("team2_id"))
    if t1 == 0 or t2 == 0:
        return None

    defense_team = t2 if offense_team == t1 else t1
    off_lin, def_lin = _get_lineups(poss, offense_team, t1, t2)

    points = fga = fga3 = ftm = fta = off_reb = def_reb = turnovers = 0
    period = _sint(poss[0].get("period"))

    for e in poss:
        a    = e.get("actionType", "")
        s    = e.get("subType", "") or ""
        r    = e.get("shotResult", "") or ""
        team = _sint(e.get("teamId"))

        if a in ("2pt", "3pt") and team == offense_team:
            fga += 1
            if a == "3pt":
                fga3 += 1
            if r == "Made":
                points += 3 if a == "3pt" else 2

        elif a == "freethrow" and team == offense_team:
            # FTs shot by the offense = real FTs (not technical FTs by opponent)
            fta += 1
            if r == "Made":
                ftm += 1
                points += 1

        elif a == "rebound":
            if s == "offensive" and team == offense_team:
                off_reb += 1
            elif s == "defensive" and team == defense_team:
                def_reb += 1

        elif a == "turnover" and team == offense_team:
            turnovers += 1

    complete = all(p > 0 for p in off_lin + def_lin)

    return {
        "offensePlayer1Id": off_lin[0],
        "offensePlayer2Id": off_lin[1],
        "offensePlayer3Id": off_lin[2],
        "offensePlayer4Id": off_lin[3],
        "offensePlayer5Id": off_lin[4],
        "defensePlayer1Id": def_lin[0],
        "defensePlayer2Id": def_lin[1],
        "defensePlayer3Id": def_lin[2],
        "defensePlayer4Id": def_lin[3],
        "defensePlayer5Id": def_lin[4],
        "points":          points,
        "possessions":     1,
        "fga":             fga,
        "fga3":            fga3,
        "ftm":             ftm,
        "fta":             fta,
        "off_reb":         off_reb,
        "def_reb":         def_reb,
        "turnovers":       turnovers,
        "gameId":          game_id,
        "period":          period,
        "offenseTeamId":   offense_team,
        "defenseTeamId":   defense_team,
        "lineup_complete": int(complete),
    }


# ── Game parser ───────────────────────────────────────────────────────────────

def parse_game(game_df: pd.DataFrame) -> list[dict]:
    """
    Parse one game's events into one-way possession rows.

    Possession-ending conditions (cdnnba format):
      • Turnover
      • Defensive rebound
      • Made field goal (unless And-1)
      • Made last free throw (unless technical FT)
      • Period end
    """
    events_raw = game_df.sort_values("actionNumber").to_dict("records")
    game_id = _sint(events_raw[0].get("gameId")) if events_raw else 0

    # Enrich each event with current lineup
    events = _build_game_lineups(events_raw)

    rows: list[dict] = []
    current: list[dict] = []   # events accumulating in this possession

    for i, e in enumerate(events):
        a = e.get("actionType", "")
        s = e.get("subType", "") or ""

        # Skip entirely
        if a in _SKIP_ACTIONS:
            continue
        # Period start: never included in a possession
        if a == "period" and s == "start":
            continue
        # Substitutions update lineup (already done in _build_game_lineups)
        # but are not possession events
        if a == "substitution":
            continue

        current.append(e)

        if _ends_possession(i, events):
            row = _build_possession_row(current, game_id)
            if row:
                rows.append(row)
            current = []

    # Flush any trailing open possession (e.g. data ends mid-period)
    if current:
        row = _build_possession_row(current, game_id)
        if row:
            rows.append(row)

    return rows


# ── Season parser ─────────────────────────────────────────────────────────────

def parse_season(
    season: int,
    playoffs: bool = False,
    output: Path | None = None,
) -> pd.DataFrame:
    label = "playoffs" if playoffs else "regular season"
    print(f"\nParsing WNBA {label} {season}...")
    raw, fmt = load_wnba_pbp(season, playoffs=playoffs)
    print(f"  Source: {fmt}")

    if fmt == "datanba":
        raw = normalize_datanba(raw)
    elif fmt == "nbastats":
        raw = normalize_nbastats(raw)

    all_rows: list[dict] = []
    game_ids = raw["gameId"].unique()
    n = len(game_ids)

    for idx, gid in enumerate(game_ids, 1):
        game_df = raw[raw["gameId"] == gid]
        game_rows = parse_game(game_df)
        all_rows.extend(game_rows)
        print(f"  [{idx:3d}/{n}] game {gid}: {len(game_rows):4d} possessions", end="\r")

    print()
    result = pd.DataFrame(all_rows)
    if len(result):
        result["season"] = season
        result["seasonType"] = "playoffs" if playoffs else "regular"
    complete = result["lineup_complete"].sum() if len(result) else 0
    print(f"  Total: {len(result):,} possessions  ({complete:,} with complete lineups)")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"  Saved → {output}")

    return result


def list_available_wnba_seasons() -> dict[str, list[int]]:
    """Return {format: [seasons]} of all WNBA PBP available in the repo."""
    lookup = _fetch_list()
    out: dict[str, list[int]] = {"cdnnba": [], "datanba": [], "nbastats": []}
    for k in lookup:
        if k.startswith("wnba_cdnnba_") and "_po_" not in k:
            try: out["cdnnba"].append(int(k.rsplit("_", 1)[-1]))
            except ValueError: pass
        elif k.startswith("wnba_datanba_") and "_po_" not in k:
            try: out["datanba"].append(int(k.rsplit("_", 1)[-1]))
            except ValueError: pass
        elif (k.startswith("wnba_nbastats_") and "_po_" not in k
              and "v3" not in k):
            try: out["nbastats"].append(int(k.rsplit("_", 1)[-1]))
            except ValueError: pass
    out["cdnnba"].sort()
    out["datanba"].sort()
    out["nbastats"].sort()
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Parse WNBA PBP into RAPM-ready possession CSV"
    )
    ap.add_argument(
        "--season", type=int, nargs="+", default=None,
        help="WNBA season start year(s), e.g. 2024. Multiple allowed.",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="Parse every WNBA season available (regular season).",
    )
    ap.add_argument(
        "--playoffs", action="store_true",
        help="Download playoff data instead of regular season.",
    )
    ap.add_argument(
        "--include-playoffs", action="store_true",
        help="With --all: also parse each season's playoffs.",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="List available WNBA seasons in the source repo and exit.",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="Combined output CSV path (concatenated across seasons).",
    )
    args = ap.parse_args()

    if args.list:
        avail = list_available_wnba_seasons()
        print("Available WNBA PBP seasons (from shufinskiy/nba_data):")
        print(f"  cdnnba   (best):     {avail['cdnnba']}")
        print(f"  datanba  (good):     {avail['datanba']}")
        print(f"  nbastats (older):    {avail['nbastats']}")
        all_yrs = sorted(set(avail["cdnnba"] + avail["datanba"] + avail["nbastats"]))
        print(f"  Combined coverage:   {all_yrs}")
        return

    if args.all:
        avail = list_available_wnba_seasons()
        seasons = sorted(set(avail["cdnnba"] + avail["datanba"] + avail["nbastats"]))
    elif args.season:
        seasons = args.season
    else:
        seasons = [2024]

    frames: list[pd.DataFrame] = []
    for season in seasons:
        # Regular season
        out = None if (args.output or args.all) else Path(
            f"data/wnba_possessions{'_po' if args.playoffs else ''}_{season}.csv"
        )
        df = parse_season(season, playoffs=args.playoffs, output=out)
        if len(df):
            frames.append(df)

        # Playoffs (when --include-playoffs)
        if args.all and args.include_playoffs:
            try:
                df_po = parse_season(season, playoffs=True, output=None)
                if len(df_po):
                    frames.append(df_po)
            except Exception as e:
                print(f"  (no playoff data for {season}: {e})")

    if frames and (args.output or args.all):
        combined = pd.concat(frames, ignore_index=True)
        out_path = args.output or Path("data/wnba_possessions_all.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        print(
            f"\nCombined {len(combined):,} possessions "
            f"({combined['lineup_complete'].sum():,} complete lineups) → {out_path}"
        )


if __name__ == "__main__":
    _cli()
