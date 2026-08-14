# WNBA RAPM — Methodology

Regularized Adjusted Plus/Minus for the WNBA, 2009–present, with a six-factor
decomposition, a career-decay variant, and a box-score estimator (DRE).

---

## 1. Data Pipeline

### Sources

| Era | Source | Format |
|---|---|---|
| 2009–2016 | shufinskiy/nba_data `wnba_nbastats_*` | NBA Stats PBP (EVENTMSGTYPE codes) |
| 2017–2021 | shufinskiy/nba_data `wnba_datanba_*` | data.nba.com |
| 2022–2025 | shufinskiy/nba_data `wnba_cdnnba_*` | cdn.nba.com |
| Current season | sportsdataverse/wehoop-wnba-data | ESPN parquet (refreshed ~daily) |

The first three are NBA-native and share player IDs. The current season comes from
ESPN because the NBA archive only publishes completed seasons and `stats.nba.com`
rate-limits per-game pulls; ESPN publishes the whole season as one parquet with no
throttling. ESPN IDs are crosswalked back to NBA IDs (see §1.3).

### 1.1 Lineup reconstruction

Play-by-play gives events, not who is on the floor. Two passes per period:

**Backward pass — infer starters.** For each player, classify their *first* event
in the period:
- first event is a **sub-out** → they were already on the floor → starter
- first event is a **non-sub action** (shot, rebound, foul) → on the floor → starter
- first event is a **sub-in** → came from the bench → not a starter

**Forward pass — track substitutions.** Seed each period with the inferred starters,
then apply subs in order to maintain a live 5-man lineup per team.

Result: **~95–96% of possessions** have all 10 players identified. Incomplete
possessions are dropped before regression (`lineup_complete == 0`).

### 1.2 Possession construction

One row per possession, with the 10 on-court player IDs plus points, FGA, FGA3, FTA,
FTM, offensive/defensive rebounds, turnovers, game/period/season, and both team IDs.

A possession ends on: turnover, defensive rebound, made FG (unless And-1), made final
free throw (unless technical), or period end.

### 1.3 ESPN → NBA ID crosswalk (current season only)

ESPN uses its own player and team IDs. The crosswalk:
1. **Exact match** on normalized name (lowercase, accents stripped, punctuation and
   Jr./Sr./III suffixes removed) against the existing NBA name table.
2. **Manual overrides** for returning veterans whose name changed (marriage,
   hyphenation dropped, nickname vs legal name). These are hand-verified — an
   unverified fuzzy match would silently split or merge careers.
3. **New players** get a synthetic ID (`90,000,000 + espn_id`), placed above the NBA
   player-ID range and below the 10-digit team-ID range so it cannot collide.

Fuzzy matching is deliberately *not* automated: shared surnames (e.g. two unrelated
Williamses) produce high similarity scores but are different people. Merging them
would corrupt multi-year windows.

---

## 2. Core RAPM Model

### Design matrix

**2N columns** — one offense column and one defense column per player, both entered
as `+1` when that player is on the floor (the Jerry Engelmann formulation):

```
column j      = +1 if player j is on offense this possession
column N + j  = +1 if player j is on defense this possession
```

Offense and defense are estimated **simultaneously in one model**, so each side is
opponent-adjusted by construction.

Because both entries are positive, sign carries the meaning:
```
ORAPM = coef[j]
DRAPM = −coef[N + j]     ← negated: a good defender suppresses opponent scoring
RAPM  = ORAPM + DRAPM
```

> **Note on encoding.** An earlier variant used `−1` for defense columns *and* negated
> the coefficient. That double-negation inverts DRAPM — elite defenders score as
> negative. If you modify the encoding, verify against known defenders before trusting
> output.

### Target and fit

```
y      = (points / possessions − league mean) × 100      # centered pts per 100
weight = possessions × 2^(−days_ago / 700)               # see §3
model  = RidgeCV(alphas=[1500 … 4000], cv=5, fit_intercept=True)
```

Alpha lands at **~1500–2250**, independently confirmed by held-out year-over-year
prediction (optimum ≈ 2000). Minimum 50 weighted possessions to qualify.

---

## 3. Time Decay

Every possession is weighted by `2^(−days_ago / 700)` relative to a reference date —
an exponential decay of the form `β^t`, with **β ≈ 0.9990**.

β was validated empirically (`scripts/td_rapm/optimize_td_rapm_beta.py`): for each test
year, fit on all prior seasons and predict that season's possessions out-of-sample.

| β | Half-life | Mean out-of-sample R² |
|---|---|---|
| 0.9990 | **693 d** | **+0.00195** ← optimal |
| 0.9993 | 990 d | +0.00194 |
| 0.9995 | 1386 d | +0.00190 |
| 0.9985 | 462 d | +0.00188 |
| 1.0000 | ∞ (no decay) | +0.00159 |
| 0.9970 | 231 d | +0.00152 |

Two findings: decay genuinely helps (every decayed β except the most aggressive beats
no-decay), and the optimum is **far slower than a fast NBA-scale decay** (β≈0.99, ~69-day
half-life). Smaller WNBA samples need more historical pooling. The peak is broad
(460–1400 d all near-optimal), so the real risk is decaying *too fast*.

Dates are real for 2017–present; 2009–2016 fall back to a season-midpoint
approximation (those games carry negligible weight anyway).

---

## 4. Six-Factor Decomposition

Six **independent one-sided regressions** on the same possessions, each with only
offense-side or only defense-side columns:

| Factor | Target |
|---|---|
| `off_ts` / `def_ts` | TS% = pts / (2 × (FGA + 0.44 × FTA)), weighted by shot volume × decay |
| `off_tov` / `def_tov` | turnovers per 100 possessions |
| `off_reb` / `def_reb` | offensive / defensive rebounds per 100 possessions |

Sign conventions are normalized so **positive is always good** (e.g. `off_tov_rapm` is
negated, since fewer turnovers is better).

### Second-stage reconstruction

A Ridge regression maps the six factors back onto RAPM:

```
RAPM_i ≈ β₁·oTS + β₂·dTS + β₃·oTOV + β₄·dTOV + β₅·oREB + β₆·dREB
```

Fit on players with ≥100 possessions, weighted by possessions, no intercept
(factors are already mean-centered). The β weights are **unit conversions** — they
translate each factor from its native units into points per 100 possessions,
producing the `*_pts` columns.

Typical: β_ts ≈ 200–350 (TS% is a 0–1 rate), β_tov ≈ 1–2, β_reb ≈ 0.5–1.

**Stage-2 R² is typically 0.90–0.95**, meaning the six factors explain ~90–95% of RAPM
variance. The residual is impact that doesn't land in any of the six channels.

### Interpreting the factors

These are **outputs, not inputs** — where impact lands, not how it was created.
A great screener creates open shots → that shows up as `oTS`. A rim protector forces
misses → `dTS`. Whatever a player does, the value registers in one of six channels.
The framework captures the *what*; the *how* is still basketball.

### Derived columns

```
o_poss_val = off_tov_val + off_reb_val     # possession control only
d_poss_val = def_tov_val + def_reb_val
poss_val   = o_poss_val + d_poss_val       # net possession margin
```

TS% is deliberately **excluded** from `poss_val`: turnovers and rebounds determine
*who gets the ball*, while TS% measures what you do with a possession you already have.

---

## 5. Career-Decay Variant

The windowed model uses fixed 1Y–5Y buckets. The career-decay variant
(`scripts/td_rapm/wnba_rapm_td.py`) instead uses a player's **entire history** with
decay as the only weighting — no arbitrary cutoff.

| | Windowed | Career-decay |
|---|---|---|
| Data | Fixed 1Y–5Y bucket | All seasons |
| Output | Row per player × year × window | One row per player |
| Reference | Oct 1 of end year | Any as-of date |
| Retired players | Present in their era | Fade out naturally |
| Answers | "What happened in this span" | "Who is best right now" |

Same core model, same six factors, same second stage. Team is assigned from each
player's **most recent game**, not their career-weighted most-common team (which would
report a traded player's former team).

---

## 6. DRE — Daily RAPM Estimate

A box-score linear-weights metric, following Kevin Ferrigan's method (Nylon Calculus,
2015 + 2017 update): regress per-100-possession box rates against multi-year RAPM, then
use the fitted coefficients as weights.

**Fit once, apply anywhere.** RAPM is needed only to learn the weights. After that DRE
is a pure box-score formula — it scores players with too few minutes for a RAPM
estimate, and works on a single game the moment the box score posts.

### Setup

- **Target**: 2-year windowed RAPM (validated, see below)
- **Inputs**: per-100 rates for PTS, FG2A, FG3A, FTA, ORB, DRB, AST, STL, BLK, TOV, PF
  (FGA split into 2PA/3PA and TRB split into ORB/DRB, per the 2017 update)
- **Weight**: season possessions
- **Fit**: weighted OLS, then scaled so PTS = 1.00

Possessions are estimated within the box score itself:
```
team_poss     = FGA − ORB + TOV + 0.44 × FTA         (team totals, that game)
player_poss   = team_poss × player_min / (team_min / 5)
```
The `/5` matters: `team_min` sums all five on-court players (≈200 for a 40-minute game),
so dividing by it directly inflates every rate ~5×.

### Choice of target window — validated, not assumed

| Window | In-sample R² | Out-of-sample r | RMSE |
|---|---|---|---|
| 1Y | 0.320 | 0.5695 | **1.663** |
| **2Y** | 0.374 | **0.5734** ← best | 1.746 |
| 3Y | 0.384 | 0.5584 | 1.905 |
| 4Y | 0.403 | 0.5550 | 2.025 |
| 5Y | 0.417 | 0.5524 | 2.053 |

**In-sample R² rises monotonically with window length — this is a smoothing artifact,
not skill.** Longer windows are less noisy and therefore mechanically easier to fit.
Selecting on R² alone would wrongly pick 5Y.

The decisive test is out-of-sample: fit weights on prior seasons, then check how well
the resulting DRE predicts a season's *actual* 1Y RAPM. There **2Y wins**, and
performance declines from 3Y on — longer windows wash out the season-specific signal
DRE exists to estimate.

### Resulting weights

```
DRE = PTS − 0.840×FG2A − 0.754×FG3A − 0.323×FTA + 0.406×ORB + 0.230×DRB
      + 0.922×AST + 2.063×STL + 0.728×BLK − 1.329×TOV − 0.370×PF − 8.127
```
(all rates per 100 possessions)

**Sanity check against Ferrigan's NBA weights:**

| | This (WNBA) | Ferrigan (NBA 2017) |
|---|---|---|
| FG2A | −0.84 | −0.9 |
| TOV | −1.33 | −1.4 |
| STL | +2.06 | +1.7 |
| BLK | +0.73 | +0.535 |
| Intercept | **−8.13** | **≈ −8.4** |

Independent convergence on the intercept is good evidence the replication is sound.

Two scales are emitted: `dre` (PTS=1, Game-Score-style bulk metric) and
`dre_rapm_scale` (raw regression output, directly comparable to RAPM).

---

## 7. Caveats

**No box priors.** Pure lineup signal — no box-score-informed stabilization. Low-minute
players are noisy; treat anyone under ~200 possessions as directional at best.

**RAPM is foundational, not definitive for single seasons.** Prior-enhanced metrics are
better for single-season estimates. The 3Y and 5Y windows are the stable views.
Early-season 1Y values are heavily shrunk toward zero — a 3.0 in 15 games is not worse
than a 6.0 in a full season, just measured with less certainty. Don't rank them
head-to-head.

**Lineup-level, not individual.** RAPM measures what happens to team outcomes when a
player is on the floor. Teammates and opponents are baked in; multi-year windows help
average this out but do not eliminate it.

**Patterns, not causal proof.** High `dTOV` with low `dTS` is *consistent with* a
gambling-defender hypothesis — it is not proof of one.

**DRE explains ~37% of RAPM variance.** It is a useful fast estimator, not a substitute.
Players whose value is defensive or gravitational are systematically undersold by any
box-only linear model. R² is lower than the NBA version because WNBA samples are smaller
on both sides of the regression.

**Mixed sources in the current season.** The live season is ESPN-sourced while history is
NBA-sourced. The crosswalk is verified, but when the NBA archive publishes the completed
season, re-running the native pipeline gives a cleaner authoritative version.
