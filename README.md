# WNBA RAPM — Live Site

A public WNBA RAPM stats explorer that updates itself every day. A scheduled
job pulls the latest games, recomputes RAPM, exports the site's data, and
publishes it to GitHub Pages.

**Live site: <https://GazelleDash.github.io/wnba-rapm/>**

Method details: [docs/METHODOLOGY.md](docs/METHODOLOGY.md)

---

## What the site is

A single-page static explorer living in `site/` — plain `index.html` +
`style.css` + `app.js`, no framework, no build step, no server. It loads a
few compact JSON files and does all sorting, filtering, and percentile
shading in the browser.

Two views, switched from the nav bar:

| View | Data | What it is |
|---|---|---|
| **Players** | `site/data/players.json` | Windowed RAPM — pick a season and a 1Y–5Y window |
| **Career-Decay** | `site/data/td.json` | Time-decayed RAPM, one row per player, weighted toward recent play |

Every numeric cell shows the value with its percentile underneath and is
heat-shaded by that percentile. Column visibility, `>=` / `<=` filters per
column, min-possession and team filters, search, and CSV export of the
current view are all in the control bar.

---

## How the automation works

```
GitHub Actions (daily cron, 09:00 UTC)
  → pulls the latest games (wehoop, with an ESPN fallback for anything
    wehoop hasn't caught up on yet)
  → recomputes RAPM
  → scripts/export_site_data.py  →  site/data/*.json
  → commits the refreshed CSVs + JSON back to this repo
       ↓
  → uploads site/ as a Pages artifact
  → deploy job publishes it to GitHub Pages
```

Runs on GitHub Actions' free tier (unlimited minutes on public repos) and
GitHub Pages (free for public repos) — no server to maintain, no paid tier.

The large possession-level file (~80 MB) is **not** committed — it's cached
between Action runs via GitHub Actions cache, so the repo itself only ever
holds the small final output tables plus the site JSON. That keeps the repo
(and every clone / deploy) fast, no matter how long the season runs.

---

## One-time setup

### 1. Push this folder to a GitHub repo

```bash
cd wnba_rapm_site
git init
git add .
git commit -m "Initial WNBA RAPM site"
git remote add origin https://github.com/<you>/<repo-name>.git
git branch -M main
git push -u origin main
```

The repo should be **public** — GitHub Pages is free for public repos.

### 2. Enable GitHub Pages (do this once)

On github.com: **Settings → Pages → Build and deployment → Source:
`GitHub Actions`**.

Do *not* pick "Deploy from a branch" — the workflow publishes via the
official `actions/deploy-pages` flow, which only works with the
`GitHub Actions` source. Until Pages is enabled this way, the `deploy` job
fails with a "Pages is not enabled" error; every other step still runs
normally.

Once enabled, the site is served at `https://<you>.github.io/<repo-name>/` —
for this repo, <https://GazelleDash.github.io/wnba-rapm/>.

### 3. Confirm the daily update is running

The workflow is in `.github/workflows/update_rapm.yml` and runs
automatically once pushed. Check progress under the **Actions** tab on
GitHub. Each successful run redeploys the site.

**The very first run is slow** (roughly 1–2 hours) — it parses the entire
season history once. Every run after that only pulls new games and takes a
few minutes. This repo ships with data pre-loaded, so the site works
immediately; the first automated run just refreshes it.

To trigger a run immediately instead of waiting for the schedule: **Actions**
tab → **Update WNBA RAPM** → **Run workflow**.

---

## Files

| Path | Purpose |
|---|---|
| `site/index.html` | The static explorer — the thing published to Pages |
| `site/style.css` | Table styling |
| `site/app.js` | Sorting, filtering, percentiles, heat shading, CSV export |
| `site/data/*.json` | `players` / `td` / `meta`, regenerated daily |
| `scripts/export_site_data.py` | Turns the pipeline CSVs into `site/data/*.json` |
| `.github/workflows/update_rapm.yml` | Daily refresh + Pages deploy |
| `scripts/` | The RAPM pipeline the workflow runs |
| `data/wnba_rapm_dashboard_v6.csv` | Windowed RAPM (1Y–5Y), refreshed daily |
| `data/td_rapm/wnba_rapm_td.csv` | Career-decay snapshot, refreshed daily |
| `docs/METHODOLOGY.md` | Full method writeup |

---

## Running the pipeline yourself (optional)

Not needed for the hosted site — it's automatic. But if you want to run it
locally:

```bash
pip install -r requirements.txt
python3 scripts/wnba_pbp_parser.py --all      # full history, ~1-2h, one time
python3 scripts/update_from_wehoop.py         # daily refresh
python3 scripts/export_site_data.py           # rebuild site/data/*.json
```

To preview the static site locally, serve `site/` over HTTP (opening
`index.html` via `file://` will block the JSON fetches):

```bash
python3 -m http.server 8000 --directory site   # then open http://localhost:8000
```

## Data sources

- [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) — historical
  WNBA play-by-play
- [sportsdataverse/wehoop-wnba-data](https://github.com/sportsdataverse/wehoop-wnba-data)
  — current-season play-by-play, refreshed daily
- ESPN's public play-by-play API — used to fill in any games wehoop hasn't
  ingested yet
