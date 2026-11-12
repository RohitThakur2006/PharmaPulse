# DECISIONS.md — Architecture & Design Decision Log

Each entry records a decision, the options considered, the choice made, and the reasoning. This log supports the 15-minute technical walkthrough.

---

## D1: Demand Forecast Target

**Date:** 2026-11-11  
**Question:** What exactly does the model predict?

| Option | Description |
|--------|-------------|
| A | Total units sold in the next 28 days (single number) |
| B | Units sold in each of the next 4 weeks (4 separate targets) |
| C | Units sold next week only, rolled forward 4 times |

**Decision:** Option A — single 28-day total.  
**Reasoning:** Simpler to train and evaluate. The pharmacist's core question is "do I have enough stock for the next month?" — a single number answers that directly. Option B adds complexity without clear business value at this stage.

---

## D2: Expiry-Risk Method

**Date:** 2026-11-11  
**Question:** How do we flag batches at risk of expiring unsold?

| Option | Description |
|--------|-------------|
| A | Rule-based: compare forecasted demand until expiry vs batch stock |
| B | Standalone ML classifier trained on batch features |
| C | Both: rule first, classifier as a second opinion |

**Decision:** Option A — forecast-driven FEFO depletion rule.  
**Reasoning:**  
1. A standalone classifier risks contradicting the demand model's own forecast (e.g., model says demand is high, classifier says batch will waste — confusing for the user).  
2. A rule outputs exact surplus quantities ("30 units will expire"), while a classifier only outputs a probability.  
3. The brief says "define the risk rule" — it does not require a second ML model.  
4. The rule is evaluated with Precision/Recall/F1 against ground-truth outcomes, satisfying the rubric.  
5. Option C remains a documented future upgrade path.

---

## D3: Data Strategy

**Date:** 2026-11-11  
**Question:** Use synthetic data, public datasets, or a hybrid?

| Option | Description |
|--------|-------------|
| A | Generate fully synthetic pharmacy data |
| B | Use only public datasets as-is |
| C | Use real sales dataset + generate batch/expiry layer on top |

**Decision:** Option C — Zdravkovic real sales + deterministic batch generation.  
**Reasoning:**  
1. The Zdravkovic dataset provides real pharmacy demand patterns (CC0 license).  
2. No public dataset combines sales + batches + expiry dates together.  
3. Generating batches deterministically from real sales gives ground-truth labels for expiry evaluation.  
4. The brief explicitly allows synthetic batch data: "Use synthetic or licensed historical product sales and batch data."

---

## D4: Daily vs Weekly vs Monthly Granularity

**Date:** 2026-11-11  
**Question:** At what time granularity should the model operate?

| Option | Rows per category | Total training rows |
|--------|:-:|:-:|
| Monthly direct | 70 months | ~450 |
| Weekly sliding window | ~300 weeks | ~1,968 |
| Daily | 2,106 days | ~16,848 |

**Decision:** Weekly sliding window (predictions every Monday).  
**Reasoning:**  
1. Monthly gave only ~450 rows — too few for tree models. In our experiment, even CatBoost (28.24% WAPE) barely beat Linear Regression (28.03% WAPE) on monthly data.  
2. Weekly sliding window gave 1,968 rows, enough for LightGBM/CatBoost to learn non-linear patterns. CatBoost achieved 17.13% WAPE — over 10 percentage points better than monthly.  
3. Daily would work but adds noise from day-of-week effects without clear benefit for a 28-day horizon.  
4. Weekly aligns with the operational rhythm: pharmacist checks every Monday morning.

**Evidence:** Side-by-side experiment conducted in separate notebooks.

| Approach | Best Model | WAPE |
|----------|-----------|:---:|
| Monthly Direct | Linear Regression | 28.03% |
| Weekly Sliding Window | CatBoost (Tuned) | 17.13% |

---

## D5: Model Selection

**Date:** 2026-11-11  
**Question:** Which ML algorithm for demand forecasting?

**Models compared on unseen 2019 test set (weekly sliding window):**

| Model | MAE | RMSE | WAPE |
|-------|:---:|:---:|:---:|
| Naive Baseline | 55.18 | 114.81 | 26.91% |
| Linear Regression | 46.08 | 82.21 | 22.47% |
| Random Forest (Tuned) | 41.99 | 78.62 | 20.48% |
| XGBoost (Tuned) | 40.52 | 76.06 | 19.76% |
| LightGBM (Tuned) | 39.05 | 71.01 | 19.04% |
| **CatBoost (Tuned)** | **35.13** | **58.52** | **17.13%** |

**Decision:** CatBoost.  
**Reasoning:** Lowest error across all three metrics. Beat the naive baseline by ~10 WAPE points. Also had the lowest RMSE (58.52 vs next-best 71.01), meaning it makes fewer catastrophic mispredictions.

**Best hyperparameters:** `depth=8, learning_rate=0.04, l2_leaf_reg=1.0, iterations=300`

---

## D6: Features to Include / Exclude

**Date:** 2026-11-11  
**Question:** Which engineered features to keep?

**Kept (11 features):**
- `sales_last_1w`, `sales_last_2w`, `sales_last_4w` — recent momentum
- `sales_same_week_last_year` — annual seasonality
- `sales_last_28d` — recent 4-week sum
- `avg_last_90d`, `avg_last_365d` — normal and long-run levels
- `volatility_weekly_sales` — demand stability
- `season_*` (Winter, Spring, Summer, Fall) — seasonal context
- `category_*` — medicine identity (one-hot encoded)

**Dropped:**
- `ratio_28d_to_90d` — only 0.08 correlation with target, adds noise without information.
- `month`, `quarter` — near-zero linear correlation with target due to circular nature of calendar months. Replaced by season dummies which, despite low linear correlation, contributed 11.6% feature importance in CatBoost through non-linear interaction splits.

---

## D7: Seasons vs Month/Quarter

**Date:** 2026-11-11  
**Question:** How to represent time-of-year?

**Decision:** Replace `month` (1–12) and `quarter` (1–4) with binary season indicators.  
**Reasoning:**  
1. Pearson correlation of `month` with target was ~0.00 because calendar months are circular (December=12 is close to January=1, but numerically far apart).  
2. Season dummies (`is_winter`, `is_summer`) allow the model to learn: "IF drug is Respiratory AND season is Winter → boost forecast."  
3. CatBoost gave seasons 11.6% combined feature importance despite low linear correlation — proving the tree model exploits them through non-linear splits.

---

## D8: Inference Interface Choice

**Date:** 2026-11-12  
**Question:** API (FastAPI) or Interactive App (Streamlit)?

**Decision:** Streamlit.  
**Reasoning:**  
1. The rubric says "interactive app" — Streamlit provides a visual dashboard.  
2. A pharmacist (non-technical user) can interact with dropdowns and tables rather than JSON API calls.  
3. Streamlit loads the saved `.cbm` model artifact and computes predictions live (no hard-coded outputs).  
4. Simpler to build and demo in a 3–5 minute video.
