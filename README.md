# WNBA RAPM — Live Site

A public WNBA RAPM dashboard that updates itself every day. Nothing to upload,
nothing to run by hand — a scheduled job pulls the latest games, recomputes
RAPM, and pushes the refreshed numbers; the site just displays whatever is
currently in the repo.

Method details: [docs/METHODOLOGY.md](docs/METHODOLOGY.md)

---

## How the automation works

```
GitHub Actions (daily cron)
  → pulls yesterday's games from wehoop
  → recomputes RAPM for the current season
  → commits the refreshed CSVs back to this repo
       ↓
Streamlit Community Cloud
  → watches this repo
  → redeploys automatically whenever it changes
```

Two free services, no server to maintain, no paid tier required.

The large possession-level file (~80 MB) is **not** committed — it's cached
between Action runs via GitHub Actions cache, so the repo itself only ever
holds the small final output tables. That keeps the repo (and every clone /
Streamlit redeploy) fast, no matter how long the season runs.

---

## One-time setup

### 1. Push this folder to a new GitHub repo

```bash
cd wnba_rapm_site
git init
git add .
git commit -m "Initial WNBA RAPM site"
```

Create an empty repo on github.com (public — Streamlit Community Cloud's free
tier needs a public repo), then:

```bash
git remote add origin https://github.com/<you>/<repo-name>.git
git branch -M main
git push -u origin main
```

### 2. Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
2. **New app** → pick this repo → main branch → main file: `streamlit_app.py`.
3. Deploy. You'll get a URL like `https://<something>.streamlit.app` —
   that's what you send your friends.

### 3. Confirm the daily update is running

The workflow is already in `.github/workflows/update_rapm.yml` and runs
automatically once pushed — no extra setup needed. Check progress under the
**Actions** tab on GitHub.

**The very first run is slow** (roughly 1–2 hours) — it parses the entire
season history once. Every run after that only pulls new games and takes a
few minutes. If you don't want to wait, this repo already ships with today's
data pre-loaded, so the site works immediately; the first automated run just
refreshes it.

To trigger a run immediately instead of waiting for the schedule: **Actions**
tab → **Update WNBA RAPM** → **Run workflow**.

---

## Adjusting the schedule

Edit the `cron` line in `.github/workflows/update_rapm.yml`
(times are UTC):

```yaml
schedule:
  - cron: "0 9 * * *"   # currently: 9am UTC daily
```

---

## Files

| Path | Purpose |
|---|---|
| `streamlit_app.py` | The dashboard — pure viewer, no computation |
| `.github/workflows/update_rapm.yml` | Daily automation |
| `scripts/` | The RAPM pipeline the workflow runs |
| `data/wnba_rapm_dashboard_v6.csv` | Windowed RAPM (1Y–5Y), refreshed daily |
| `data/td_rapm/wnba_rapm_td.csv` | Career-decay snapshot, refreshed daily |
| `data/dre/` | Box-score DRE estimate, refreshed daily |
| `docs/METHODOLOGY.md` | Full method writeup |

---

## Running the pipeline yourself (optional)

Not needed for the hosted site — it's automatic. But if you want to run it
locally:

```bash
pip install -r requirements.txt
python3 scripts/wnba_pbp_parser.py --all      # full history, ~1-2h, one time
python3 scripts/update_from_wehoop.py         # daily refresh
python3 -m streamlit run streamlit_app.py     # view locally
```

## Data sources

- [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) — historical
  WNBA play-by-play
- [sportsdataverse/wehoop-wnba-data](https://github.com/sportsdataverse/wehoop-wnba-data)
  — current-season play-by-play and box scores, refreshed daily
