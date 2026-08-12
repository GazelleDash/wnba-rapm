"""
WNBA RAPM — public dashboard.

Reads the CSVs the GitHub Actions workflow refreshes daily
(.github/workflows/update_rapm.yml) and renders them. No computation happens
here — this file is purely a viewer.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

DASH_V6  = Path("data/wnba_rapm_dashboard_v6.csv")
TD_RAPM  = Path("data/td_rapm/wnba_rapm_td.csv")

st.set_page_config(page_title="WNBA RAPM", page_icon="🏀", layout="wide")


@st.cache_data(ttl=3600)
def load_windowed() -> pd.DataFrame | None:
    return pd.read_csv(DASH_V6) if DASH_V6.exists() else None


@st.cache_data(ttl=3600)
def load_td() -> pd.DataFrame | None:
    return pd.read_csv(TD_RAPM) if TD_RAPM.exists() else None


def last_updated() -> str:
    if DASH_V6.exists():
        ts = os.path.getmtime(DASH_V6)
        return pd.Timestamp(ts, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    return "unknown"


st.title("🏀 WNBA RAPM")
st.caption(
    f"Regularized Adjusted Plus/Minus, 2009–present · six-factor breakdown · "
    f"auto-updated daily · last refresh **{last_updated()}**"
)

tab_windowed, tab_decay, tab_about = st.tabs(
    ["Windowed RAPM", "Career-Decay (live)", "About"]
)

# ── Windowed RAPM ────────────────────────────────────────────────────────────
with tab_windowed:
    df = load_windowed()
    if df is None:
        st.info("No data yet — the daily pipeline hasn't run. Check back after "
                 "the next scheduled update, or trigger it from the Actions tab.")
    else:
        years = sorted(df["end_year"].unique(), reverse=True)
        default_year_idx = years.index(2026) if 2026 in years else 0

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            end_year = st.selectbox("Season", years, index=default_year_idx)
        with c2:
            window = st.selectbox(
                "Window", sorted(df["rapm_length"].unique()), index=0,
                format_func=lambda w: f"{w}Y")
        with c3:
            min_poss = st.slider(
                "Minimum possessions", 0, 3000, 300, step=50,
                help="Filter out small samples — early-season and deep-bench "
                     "numbers are noisy.")

        sub = df[(df["end_year"] == end_year) & (df["rapm_length"] == window)]
        sub = sub[sub["total_poss"] >= min_poss].sort_values("rapm", ascending=False)

        st.dataframe(
            sub[["name", "team", "total_poss", "rapm", "orapm", "drapm",
                 "off_ts_val", "off_tov_val", "off_reb_val",
                 "def_ts_val", "def_tov_val", "def_reb_val"]],
            column_config={
                "name": "Player", "team": "Team",
                "total_poss": st.column_config.NumberColumn("Poss", format="%.0f"),
                "rapm": st.column_config.NumberColumn("RAPM", format="%.2f"),
                "orapm": st.column_config.NumberColumn("ORAPM", format="%.2f"),
                "drapm": st.column_config.NumberColumn("DRAPM", format="%.2f"),
                "off_ts_val": st.column_config.NumberColumn("oTS", format="%.2f"),
                "off_tov_val": st.column_config.NumberColumn("oTOV", format="%.2f"),
                "off_reb_val": st.column_config.NumberColumn("oREB", format="%.2f"),
                "def_ts_val": st.column_config.NumberColumn("dTS", format="%.2f"),
                "def_tov_val": st.column_config.NumberColumn("dTOV", format="%.2f"),
                "def_reb_val": st.column_config.NumberColumn("dREB", format="%.2f"),
            },
            hide_index=True, use_container_width=True, height=560,
        )

        top15 = sub.head(15).set_index("name")["rapm"].sort_values()
        st.bar_chart(top15, horizontal=True)

# ── Career-decay ─────────────────────────────────────────────────────────────
with tab_decay:
    td = load_td()
    if td is None:
        st.info("No career-decay data yet.")
    else:
        st.caption(
            "One number per player, using their ENTIRE history with "
            "exponential time decay (β ≈ 0.999, ~700-day half-life) instead "
            "of a fixed multi-year window — a DARKO-style 'who's best right "
            "now' snapshot, as of the date shown above."
        )
        min_poss_td = st.slider(
            "Minimum possessions ", 0, 3000, 300, step=50, key="td_min_poss")
        sub = td[td["total_poss"] >= min_poss_td].sort_values("RAPM", ascending=False)
        st.dataframe(
            sub[["name", "team", "total_poss", "RAPM", "ORAPM", "DRAPM"]],
            column_config={
                "name": "Player", "team": "Team",
                "total_poss": st.column_config.NumberColumn(
                    "Eff. Poss", format="%.0f",
                    help="Decay-weighted — recent possessions count near full, "
                         "old ones fade, so this is lower than a raw career total."),
                "RAPM": st.column_config.NumberColumn(format="%.2f"),
                "ORAPM": st.column_config.NumberColumn(format="%.2f"),
                "DRAPM": st.column_config.NumberColumn(format="%.2f"),
            },
            hide_index=True, use_container_width=True, height=560,
        )

# ── About ─────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
### What this is

Regularized Adjusted Plus/Minus for the WNBA, built from public play-by-play
data (2009–present). RAPM estimates a player's impact on scoring margin per
100 possessions, adjusted for teammates and opponents via ridge regression on
lineup data.

**Reading the numbers:** `+5.0` means a team is roughly 5 points per 100
possessions better with that player on the floor. Filter by possessions —
low-minute samples are noisy. Early-season numbers within a season are
shrunk toward zero by design; they get more stable as the season fills out.

**Data refreshes automatically every day** via GitHub Actions — no one is
updating this by hand.

Full method writeup, caveats, and validation results: see `docs/METHODOLOGY.md`
in the repo.

**Sources:** [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data)
(historical play-by-play) ·
[sportsdataverse/wehoop-wnba-data](https://github.com/sportsdataverse/wehoop-wnba-data)
(current season, refreshed daily)
        """
    )
