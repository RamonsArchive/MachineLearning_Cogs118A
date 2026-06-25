# Experiment 1 — ModalAI Part/Component Demand Forecasting (Regression)

> **UPDATE — now on the v2 panel.** Experiment 1 uses `part_week_panel_v2.csv`, which folds
> the **entire Shopify signal** into `demand_qty` (BOM roll-down of finished-good sales to
> components + direct part-matched sales). This extends the series from 29 → **47
> demand-bearing weeks (70-week span)** and adds leak-safe predictors (`bom_indegree`,
> `wo_opened_lag1/roll4`, `weeks_since_last_demand`; `engineer_id`/`demand_adi` dropped for
> ≥80% null). The §1/§4 history below describes the original v1 starvation problem and how it
> was mitigated. A sibling **Experiment 3** (`forecast_sku_sales.py`) forecasts Shopify
> SKU-sales directly on the full 70 weeks and beats the naive baselines more decisively.

**A defense of the methodology, a full data-leakage audit, and an honest diagnosis of
what limits performance — which is mostly *not* the hyperparameters.**

This experiment plugs a new regression dataset (the ModalAI `part_week_panel`) into the
existing Caruana & Niculescu-Mizil–style harness (XGBoost / Random Forest / Neural Net,
20/50/80 splits × 3 trials, GridSearchCV, RMSE primary). Code:

- `src/experiments/forecast_demand.py` — orchestrator
- `src/utils/{load,clean,eda}/*forecast_demand*.py` — load / leakage+null policy / EDA
- `src/graphs/forecast_demand_plots.py` — per-model reports + comparison
- Outputs: `src/results/forecast_demand_all_results.json`, `plots/forecast_demand_plots/...`

---

## TL;DR

The pipeline is methodologically sound and leakage-controlled for the features it *trains*
on. But the headline R² is low and the train/test gap is wide **for structural,
data-level reasons, not because the grids are mistuned**:

1. **History is short, and the demand signal starts late — so we restrict to it.** The
   component demand signal (StockHistory + BuildStorage) only begins **2025-12-08**;
   **41 of 70 weeks are identically zero for every part** (verified contiguous — no interior
   gaps). On the full 70-week panel, the temporal `20_80` and `50_50` splits trained on an
   all-zero target (which also collapsed the winsor cap to 0 and faked RMSE=0 / R²=1).

   **Fix applied (`clean_forecast_demand.py`):** the panel is now restricted to
   **demand-bearing weeks (≥ 2025-12-08): 29 weeks, 52,461 rows, 1,809 parts, 13.5%
   non-zero.** Lags/rolls/Croston are computed on the full history *before* the row filter,
   so their values at the onset weeks stay correct. All three splits now train on real
   signal:

   | Split | Train weeks | Train rows (demand>0) | Test rows (demand>0) |
   |---|---|---|---|
   | `20_80` | 6 | 10,854 (1,236 = 11.4%) | 41,607 (5,850) |
   | `50_50` | 14 | 25,326 (3,309 = 13.1%) | 27,135 (3,777) |
   | `80_20` | 23 | 41,607 (5,348 = 12.9%) | 10,854 (1,738) |

   A belt-and-suspenders guard still **skips any split whose train target is all-zero**
   (`forecast_demand.py`), so the degenerate case can never reappear. Caveat: 29 weeks
   (~7 months) is enough to fit and compare models and beat baselines, but too short to
   learn seasonality; `80_20` (23 train weeks) is the most trustworthy split.

2. **Every static feature is constant within a part** (verified: `nunique == 1` per
   `part_id` for all 11 static columns). They are therefore *part-identity fingerprints*,
   not time-varying signal — and `current_stock` / `on_order_current` are *today's*
   snapshot, i.e. mild look-ahead. This inflates the train fit (memorization) and the
   train/val gap independently of tree depth.

3. **94% of part-weeks are zero** (heavy recount tail on the rest). RMSE and R² are
   dominated by a handful of large values, so a low R² here is expected even for a good
   intermittent-demand model. **The honest lens is MAE-on-non-zero-weeks and "did we beat
   the naive baselines," both of which the models do.**

The hyperparameter retune (bias↑/variance↓) only narrows the *within-training* train/val
gap. It cannot manufacture signal that the early weeks don't contain. So tuning helps the
`80_20` story a little and the other two splits essentially not at all.

---

## 1. Data-leakage audit (every predictor we use)

**Target:** `demand_qty = max(sh_outflow_qty, build_consumed_qty)` — winsorized at the
train p99.9 (recount-spike denoising) then `log1p`; all metrics inverted to real units.

### 1a. Columns we *excluded* (hard leakage — removed in `clean_forecast_demand.py`)

| Excluded column(s) | Why it leaks |
|---|---|
| `sh_outflow_qty`, `build_consumed_qty` | **Define the target** (`max` of the two). |
| `sh_net_delta`, `sh_inflow_qty`, `sh_movement_events`, `build_events`, `shopify_ordered_qty`, `shopify_unfulfilled_qty`, `on_hand_eow` | Same-week (contemporaneous) signals — known only *at/after* week *t*. |
| `part_id`, `part_name`, `mpn` | Identifiers; the SKU name encodes the project (`notes.txt`) → memorization. |
| `cum_zero_share` | **Verified leak:** built with `.expanding().mean()` over `(demand_qty==0)` which **includes week *t*** (not shifted). It tells the model whether *this* week is zero. Dropped. |
| `low_stock_level` (99.1% null), `lead_time_days` (98.2% null) | ≥80% null, non-target → dropped per null policy. |
| `kanban`, `virtual` | Single unique value (zero variance). |
| `iso_year`, `week_start` | `iso_year` is a raw time index (trivial trend across a temporal split); `week_start` is the split key, not a feature. |

### 1b. Columns we *keep* — and the residual caveat we flag

Allowed predictors (26): lag/rolling demand `demand_lag_{1,2,4,8,12}`,
`demand_roll_mean_{4,8}`, `demand_roll_std_4`; calendar `iso_week, month, quarter,
week_sin, week_cos`; static `manufacturer, status, archived, subassembly, is_project,
unit_cost_current, price_current, current_stock, bom_component_count, on_order_current,
moq, order_multiple, num_vendors`.

| Feature group | Leakage verdict | Notes |
|---|---|---|
| **Lag / rolling demand** | ✅ Leak-safe | All `.shift()`-ed in `build_god_dataset.py` (`shift(l)`, `shift(1).rolling(...)`). Computed on the full panel but only look **backward**, so global computation does not bleed future into past. |
| **Calendar** | ✅ Leak-safe | Known in advance. (Caveat: months in the test horizon may be unseen in train — a generalization limit, not leakage.) |
| `manufacturer`, `subassembly`, `is_project`, `bom_component_count` | ✅ Effectively time-invariant | A part's manufacturer / make-vs-buy / BOM size doesn't change; constant-per-part is fine here. |
| `unit_cost_current`, `price_current`, `moq`, `order_multiple`, `num_vendors` | ⚠️ **Current snapshot** | Slowly-varying vendor/cost terms taken as of the extract date, not as-of week *t*. Mild look-ahead; kept per the task spec, flagged here. |
| `current_stock`, `on_order_current`, `status`, `archived` | ⚠️ **Current snapshot — strongest residual risk** | These are *today's* on-hand / on-order / lifecycle state. The feasibility report explicitly classes `Part.stock` and `on_order` as leakage ("reconstruct as-of week start"). `archived/status` encode the part's *end-state*, which correlates with recent (test-period) zeros. **Verified constant-per-part**, so they also act as identity fingerprints. |

**Why we kept the ⚠️ columns anyway:** the task specification explicitly lists them as
allowed static predictors with the instruction *"static attrs are current snapshots (a
known caveat) — keep them but flag it."* We comply: they are kept, and flagged loudly
here. **For a rigorous backtest they should be dropped or reconstructed as-of-week-start**
(see §4). No *hard* leakage (same-week signal / target component / identifier) reaches the
models.

### 1c. Preprocessing is leakage-safe
- Winsor cap fit on **train only**, then applied to both.
- Median imputation (RF/NN variant) fit on **train only**; XGBoost keeps NaN natively.
- Temporal split: every test week is **strictly later** than every train week (no shuffle).

---

## 2. Methodology defense (the choices that are deliberate)

| Choice | Justification |
|---|---|
| **Temporal split, not shuffled** | This is a time series; a random shuffle would put future weeks in train and leak. Earliest 20/50/80% of weeks → train; the rest → test, reproducing the harness's three ratios as *time* fractions (`20_80`/`50_50`/`80_20`). |
| **Winsorize + `log1p` target** | EDA shows ±61k weekly spikes that are recount/correction artifacts, not demand, and a 94%-zero, heavy-tailed distribution. Winsorizing at train p99.9 denoises the artifacts; `log1p` stabilizes variance. Metrics are inverted to real units; predictions clipped to `[0, cap]` (an MLP's unbounded log output otherwise explodes through `expm1`). |
| **Per-algorithm null policy** | Keep null **rows** (missingness is informative; dropping rows = survivorship bias toward high-volume parts). XGBoost gets NaN natively; RF/NN get **median-impute + a `*_was_null` indicator** (so "unknown" stays learnable) + scaling. Matches the feasibility report's §6 policy. |
| **Naive baselines reported** | Under 94% zeros, a flattering R² can hide a model that loses to "predict last week." We score **last-value**, **4-week moving-average**, and **Croston** (leak-safe, computed from strictly-prior weeks) on the same held-out weeks, and judge models against that floor. |
| **MAE on non-zero part-weeks** | The business-relevant error is on the weeks that actually have demand; global RMSE is dominated by the zeros and the tail. |
| **3 trials/split** | The temporal split is deterministic, so trials vary only model init + CV-fold shuffling → small but non-zero std (reported as mean ± std). This is expected, not a bug. |

---

## 3. What the hyperparameters can and cannot do

The first smoke run overfit (XGB CV-train −0.092 vs CV-val −0.220; RF train RMSE 17.8 vs
test 60). The available **bias-variance levers** are: shallower trees (`max_depth` 3–6),
larger leaves (`min_child_weight` / `min_samples_leaf` ↑), row+column subsampling
(`subsample`, `colsample_bytree`), stronger L1/L2 (`reg_alpha`, `reg_lambda`), smaller MLP
nets with stronger weight decay (`alpha` ↑). These **narrow the within-training train/val
gap** and are worth setting toward the regularized end.

**But they do not touch the two dominant limiters** (§TL;DR #1 and #2): the short,
late-starting history (now mitigated by restricting to the 29 demand-bearing weeks) and the
constant-per-part identity/snapshot features. That is the precise sense in which *"it's not
just the hyperparameters."* Even with all three splits now carrying signal, only ~29 weeks
of history caps how well any model can generalize.

---

## 4. Recommended v2 (to actually raise the ceiling)

1. ✅ **DONE — restrict to demand-bearing weeks** (≥ 2025-12-08; 29 weeks, 52,461 rows).
   Implemented in `clean_forecast_demand.py`; all three temporal splits now train on real
   demand. Next step up would be an **expanding-window backtest** for more robust temporal CV.
2. **Drop or reconstruct the current-snapshot statics** (`current_stock`, `on_order_current`,
   `status`, `archived`, cost/price) as **as-of-week-start** values; today's snapshots are
   look-ahead and constant-per-part identity proxies.
3. **Two-stage (hurdle) model**: classify will-this-part-move? then regress how-much — the
   standard fix for 94% zero-inflation, and recommended by the feasibility report.
4. **More history**: accrue weeks; the component signal is ~6 months old, so seasonality is
   unlearnable yet.

These are data/representation fixes; they dominate any further grid search.

---

## 5. How to run

Run **from the `empirical-ml-comparison/` directory** (so EDA plots land in the canonical
`plots/`). `-u` streams progress live so you can watch `SPLIT…/Trial N/3/GRID SEARCH COMPLETE`.

```powershell
cd C:\Users\Ramon\Desktop\Developer\ML_Projects\ML_Pipeline\empirical-ml-comparison

# Fast pipeline check (1 split x 1 trial, tiny grids) — reports tag themselves [SMOKE TEST]
$env:FORECAST_SMOKE=1; python -u src/main.py

# Full run (3 splits x 3 trials) — reports tag themselves [full run]; est. ~15-35 min
Remove-Item Env:\FORECAST_SMOKE -ErrorAction SilentlyContinue
python -u src/main.py 2>&1 | Tee-Object exp1_run.log
```

Equivalently run the experiment module directly: `python -u src/experiments/forecast_demand.py`
(same outputs). Every report header now prints the actual `N split(s) x M trial(s)` and a
`[SMOKE TEST]` / `[full run]` tag, so smoke vs. real is never ambiguous.
