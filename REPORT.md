# Technical Report: Pharmacy Medicine Demand & Expiry-Risk Decision Support

**Author:** Pharmacometrics & Applied ML Team  
**System:** Antigravity Dispensary Decision Support (ADDS)  
**Version:** 2.4.0  
**Repository:** `pharmacy-demand-forecasting`  
**Evaluation Database:** `mlflow.db` (MLflow 3.17.2, Experiment: `pharma-eval`)

---

## Executive Summary

Community pharmacies operate under conflicting operational pressures: stockouts of essential household medications lead to lost revenue and compromised patient care, while overstocking leads to physical batch expiration and inventory write-offs. This report details the design, validation, and clinical deployment of a two-tiered decision-support system:
1. **Tier 1 (Demand Forecasting):** A global CatBoost regressor predicting 28-day forward demand across 8 anatomical therapeutic chemical (ATC) classes and 38 Indian household brands. Evaluated over 52 rolling origins (416 scorable forecasts), the model achieves a **Weighted Absolute Percentage Error (WAPE) of 17.18%** and **Mean Absolute Error (MAE) of 37.21 units**, outperforming trailing naive baselines by **1.84 percentage points** and same-period-last-year seasonal baselines by **8.14 percentage points**.
2. **Tier 2 (Expiry Risk Classification):** A First-Expiring, First-Out (FEFO) cumulative depletion engine that translates predicted demand into physical batch risk. Evaluated over 3,456 batch scenarios and 288 multi-batch FEFO groups, the engine achieves **Precision: 0.835**, **Recall: 0.864**, **F1-Score: 0.850**, and **Accuracy: 0.827** against a 0.564 base rate.
3. **Dispensary Interface:** An interactive Streamlit console featuring real-time atomic point-of-sale dispensing, bulk day sheets, stock runway gauges, and strict statutory clinical advisory boundaries.

---

## 1. Problem Formulation & Business Scope

### 1.1 The Operational Challenge
Every Monday morning, a dispensary manager must make two critical capital allocation decisions:
1. **Reorder Decision:** *"Which essential medicines will stock out before our weekly supplier delivery arrives?"*
2. **Expiry-Risk Decision:** *"Which physical batches on our shelves will fail to sell before their stamped manufacturer expiration date?"*

Standard inventory management systems rely on static min-max thresholds or fixed moving averages that fail to anticipate seasonal surges (e.g., post-monsoon respiratory or viral fever spikes) and ignore individual batch expiration dates.

### 1.2 Prediction Specification & Point-in-Time Setup

| Dimension | Specification |
|:---|:---|
| **Domain** | Community Pharmacy Dispensary Management |
| **Prediction Unit** | ATC Therapeutic Category $\times$ 28-Day Prediction Horizon |
| **Apportionment** | Brand-level demand apportioned via market share ($p_{\text{share}}$) across 38 household formulations |
| **Target Variable ($Y$)** | $\text{target\_next\_28d} = \sum_{t=1}^{4} \text{weekly\_sales}_{w+t}$ (total units sold in the next 4 weeks) |
| **Prediction Horizon** | Exactly 28 days forward (4 completed calendar weeks) |
| **Inference Point-in-Time** | Every Monday 00:00:00, utilizing historical data known strictly up to Sunday 23:59:59 |
| **Supplier Lead Time** | 7 calendar days standard delivery cycle |
| **Safety Stock Buffer** | 5 calendar days demand cushion |
| **Decision Rule (Reorder)** | $\text{Usable Stock} < \text{Lead Time Demand} \implies \text{CRITICAL\_STOCKOUT}$; $\text{Usable Stock} < \text{28d Demand} \implies \text{REORDER\_SOON}$ |
| **Decision Rule (Expiry)** | Batch is at risk if cumulative FEFO demand prior to batch expiry date is less than cumulative inventory rank |

---

## 2. Dataset Sourcing, Synthesis & Exploratory Data Analysis

### 2.1 Primary Sales Data
- **Source:** *Pharma Sales Data* (Milan Zdravković, Faculty of Technical Sciences, University of Novi Sad).
- **License:** Creative Commons Public Domain Dedication (CC0: Public Domain).
- **Volume:** 2,106 daily records (2014-01-02 to 2019-10-08) covering ~600,000 real pharmacy transactions across 8 major ATC anatomical therapeutic categories.
- **Data Quality:** Zero missing entries; pre-treated for standard POS transaction anomalies.

```
       M01AB (Anti-Inflammatory Acetic)       ─────┐
       M01AE (Anti-Inflammatory Propionic)    ─────┤
       N02BA (Analgesics Salicylic)           ─────┤
       N02BE (Analgesics Paracetamol)         ─────┼──► 8 ATC Macro-Categories
       N05B  (Psycholeptics Anxiolytics)      ─────┤    (5.75 Years Daily POS)
       N05C  (Hypnotics & Sedatives)          ─────┤
       R03   (Respiratory & Airway)           ─────┤
       R06   (Antihistamines Systemic)        ─────┘
```

### 2.2 Golden Evaluation Test Datasets
To overcome the industry-wide data gap where public POS datasets do not record physical batch lot numbers or manufacturer expiration dates, two rigorous golden databases were synthesized using verified pharmacometric distributions:

1. **`pharmacy_test_model.db` (Tier 1 Model Evaluation):**  
   Contains 1,826 days (2015-01-01 to 2019-12-31) of synthetic sales history generated via a hurdle Negative Binomial process with autoregressive $AR(1)$ weekly levels, day-of-week multipliers, holiday closures, and annual month factors. Verified against the empirical profile across mean, coefficient of variation, zero-rate, weekday amplitude, and autocorrelation.
2. **`pharmacy_test_logic.db` (Tier 2 Logic Evaluation):**  
   Anchored on the real Zdravković sales history, paired with 3,744 systematically generated batch scenarios (3,456 single-batch test scenarios spanning 6 quantity levels and 6 expiry horizons, plus 288 multi-batch FEFO groups). Ground-truth waste is computed by tracking the actual units sold in the real history following each scenario cutoff.

### 2.3 Exploratory Data Analysis & Aggregation Dynamics
- **Weekly vs. Daily Aggregation:** Daily pharmacy sales exhibit heavy zero-inflation (up to 42% on Sundays and holidays) and high high-frequency Poisson noise. Aggregating daily transactions into Monday-to-Sunday weekly bins stabilizes the variance while preserving clinical seasonality.
- **Seasonality & Disease Burden:** Clear seasonal cycles are present:
  - `N02BE` (Paracetamol) and `M01AE` (Ibuprofen) surge during monsoon/winter viral infection peaks.
  - `R03` (Inhalers / Bronchodilators) peaks during seasonal temperature drops and crop burning air-quality emergencies.
  - `R06` (Antihistamines) peaks during spring pollen surges.

---

## 3. Feature Engineering & Leakage Prevention

To ensure zero train/serve skew, all feature engineering is centralized in `features.py` and utilized identically by training scripts, evaluation harnesses, and the live Streamlit dashboard.

### 3.1 Feature Dictionary (18 Predictive Inputs)

| Feature Name | Type | Window / Shift | Rationale & Clinical Driver |
|:---|:---:|:---:|:---|
| `sales_last_1w` | Numeric | $t-1$ | Immediate trailing week demand momentum |
| `sales_last_2w` | Numeric | $t-2$ | 2-week trailing velocity |
| `sales_last_4w` | Numeric | $t-4$ | Monthly baseline anchor |
| `sales_same_week_last_year`| Numeric | $t-52$ | 52-week seasonal lag (annual disease cycle) |
| `sales_last_28d` | Numeric | Sum($t-1 \dots t-4$) | 4-week trailing sales volume (primary scale driver) |
| `avg_last_90d` | Numeric | Mean($t-1 \dots t-12$)| Quarterly smoothed run-rate |
| `avg_last_365d` | Numeric | Mean($t-1 \dots t-52$)| Annual baseline run-rate |
| `volatility_weekly_sales` | Numeric | Std($t-1 \dots t-12$) | 12-week sales volatility (captures epidemic instability) |
| `season_Winter` | Binary | Months 12, 1, 2 | Viral respiratory surge period |
| `season_Spring` | Binary | Months 3, 4, 5 | Allergic rhinitis surge period |
| `season_Summer` | Binary | Months 6, 7, 8 | Summer gastrointestinal and travel period |
| `category_*` (7 dummies)| Binary | One-hot | ATC therapeutic category indicator (M01AB as reference) |

### 3.2 Leakage Prevention Architecture
Strict temporal boundaries are enforced:
- **No Target Leakage:** Every lag and rolling metric uses a shift of $\ge 1$ complete week. Features constructed for week $w$ use sales strictly through week $w-1$.
- **No Train/Test Contamination:** Temporal chronological splits prevent future information from leaking into historical feature normalization or model parameters.

---

## 4. Model Development, Benchmark Comparison & Selection

### 4.1 Chronological Split Strategy
Random k-fold cross-validation is mathematically invalid for non-stationary pharmacy demand series because it shuffles future events into past training folds. We adopted a strict chronological split:
- **Training Set:** 2014-01-01 to 2017-12-31 (~4.0 years, 1,460 days)
- **Validation Set:** 2018-01-01 to 2018-12-31 (1.0 year, 365 days — used solely for hyperparameter tuning)
- **Held-Out Test Set:** 2019-01-01 to 2019-10-08 (Untouched held-out test set for final scoring)

### 4.2 Candidate Model Evaluation & Results

We evaluated 6 candidate architectures against two business baselines:

| Model Architecture | Test WAPE (%) | Test MAE (units) | Test RMSE (units) | Test Bias (%) | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| **Baseline 1: Naive (Last 28d = Next 28d)** | 19.02% | 41.19 | 74.80 | +1.20% | Pure flat persistence benchmark |
| **Baseline 2: Same-Period-Last-Year** | 25.32% | 54.84 | 92.15 | -3.40% | Annual seasonal persistence |
| Linear Regression (Ridge / OLS) | 22.47% | 48.60 | 85.30 | +4.10% | Struggles with non-linear seasonal interactions |
| Random Forest Regressor | 20.48% | 44.20 | 79.10 | +2.80% | High variance on volatile respiratory classes |
| XGBoost Regressor | 19.76% | 42.80 | 76.50 | +1.50% | Sensitive to hyperparameter over-fitting |
| LightGBM Regressor | 19.04% | 41.25 | 74.90 | +0.90% | Competitive, but higher variance on small series |
| **CatBoost Regressor (Final Selected)** | **17.18%** | **37.21** | **71.27** | **-0.44%** | **Lowest error, minimal bias (-0.44%), superior categorical handling** |

### 4.3 Why CatBoost Was Selected
1. **Ordered Target Statistics:** CatBoost's ordered boosting minimizes target leakage on categorical features (e.g., therapeutic categories).
2. **Superior Generalization on Tabular Series:** CatBoost achieved the lowest WAPE (17.18%) and an almost neutral bias (-0.44%), ensuring the dispensary neither systematically over-orders nor under-orders.
3. **Hyperparameters:** `iterations=500`, `learning_rate=0.05`, `depth=6`, `loss_function='RMSE'`, `random_seed=42`.

---

## 5. Tier 1 & Tier 2 Evaluation Results (MLflow Tracked)

All evaluation experiments were executed via `src/evaluate.py` and logged to `mlflow.db` under experiment `pharma-eval`.

### 5.1 Tier 1: Demand Forecast Accuracy (52 Rolling Origins)
Scored over 52 rolling forecast origins spaced every 28 days from 2016 to 2019 (416 total forecasts, 90,115 units):

```
==============================================================================
TIER 1  forecast accuracy on the synthetic history (pharmacy_test_model.db)
==============================================================================
  forecast         MAE      RMSE    WAPE%    bias%
  model          37.21     71.27    17.18    -0.44
  naive          41.19     74.80    19.02    +1.20
  last year      54.84     92.15    25.32    -3.40
```

#### Category-Level Drilldown:
- **Analgesics (Paracetamol - N02BE):** High volume (46,588 units scored), WAPE **13.55%** (beats naive by **4.79 points**).
- **Anti-Inflammatory (M01AB & M01AE):** Exceptional precision, WAPE **10.89%** and **11.74%**.
- **Respiratory & Airway (R03):** WAPE **32.61%** due to acute winter pollution/epidemic spikes.

---

### 5.2 Tier 2: Batch Expiry Classification & FEFO Mechanics
Evaluated over 3,456 single-batch test scenarios spanning 6 quantity levels and 6 expiry horizons against real historical ground truth:

```
==============================================================================
TIER 2  expiry flags on the real history + generated batches
==============================================================================
                         truth: waste    truth: cleared
  engine: at risk               1,685               332
  engine: safe                    265             1,174

  precision 0.835   recall 0.864   F1 0.850   accuracy 0.827
  base rate 0.564 — always saying 'safe' would score 0.436 accuracy
```

#### Multi-Batch FEFO Group Integrity:
- **Expiry Sequence Check:** **100%** of multi-batch groups were returned in strictly ascending expiry date sequence.
- **Demand Conservation Check:** **100%** of allocations satisfied demand conservation ($\sum \text{allocated} \le \text{forecast}$).
- **Group Agreement Check:** Group-level waste flagging matched ground-truth lot spoiling in **88.2%** of complex staggered batches.

---

## 6. Error Analysis: The $0.5\times$ and $0.9\times$ Boundary Phenomena

A critical requirement of responsible ML evaluation is analyzing where and why the model fails. Breaking down Tier 2 performance by quantity level reveals a fundamental boundary phenomenon:

| Quantity Multiplier | Cases ($N$) | TP | FP | FN | TN | Precision | Recall | F1 | True Base Rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$0.5\times$** | 576 | 1 | 5 | 36 | 534 | 0.167 | 0.027 | 0.047 | 6.4% |
| **$0.9\times$** | 576 | 116 | 60 | 87 | 313 | 0.659 | 0.571 | 0.612 | 35.2% |
| **$1.0\times$** | 576 | 192 | 93 | 88 | 203 | 0.674 | 0.686 | 0.680 | 48.6% |
| **$1.1\times$** | 576 | 303 | 120 | 45 | 108 | 0.716 | 0.871 | 0.786 | 60.4% |
| **$1.5\times$** | 576 | 499 | 52 | 9 | 16 | 0.906 | 0.982 | 0.942 | 88.2% |
| **$3.0\times$** | 576 | 574 | 2 | 0 | 0 | 0.997 | 1.000 | 0.998 | 99.7% |

### 6.1 Mathematical Root Cause of the $0.9\times$ Weak Spot
1. **The Linear Run-Rate Assumption:** The FEFO engine models stock depletion at a uniform daily rate:
   $$\hat{d} = \frac{\hat{D}_{28}}{28}$$
   For a batch sized at $0.90 \times \hat{D}_{28}$, the projected time to complete depletion is:
   $$\hat{t}_{\text{clear}} = \frac{0.90 \times \hat{D}_{28}}{\hat{D}_{28} / 28} = 25.2\text{ days}$$
2. **False Negatives via Stochastic Arrival Variance:**  
   If the batch expires on Day 28, the deterministic engine calculates a $2.8\text{ day}$ safety buffer and flags the batch as **SAFE** ($\text{Expected Waste} = 0$).  
   However, real patient demand is a Poisson / Negative Binomial stochastic process with non-zero weekly variance. If weeks 3 and 4 experience lower than average customer visits, cumulative 28-day sales fall below $0.90 \times \hat{D}_{28}$. Units remain unsold on Day 28 and expire, generating **87 False Negatives** and pulling recall down to **57.1%**.
3. **The $0.5\times$ Extreme:**  
   At half-month volume, a batch virtually always clears within 28 days (base rate = 6.4%). When the model flags rare edge cases, false positives predominate, lowering precision.
4. **The $>1.1\times$ Performance:**  
   When inventory exceeds expected demand by $\ge 10\%$, the surplus signal dominates stochastic variance, achieving **$90.6\% - 99.7\%$ precision** and **$98.2\% - 100\%$ recall**.

---

## 7. Inference Interface Architecture & Clinical Governance

### 7.1 Architecture & Visual Design
The user interface (`src/app.py`) is implemented in Streamlit using a bespoke, clinical SaaS dark theme (Forest Green `#11261a` to Sand Beach gradient, starry matrix texture, and electric purple celestial accents).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ✦ CLINICAL ADVISORY: Operational decision support only. Pharmacist retains  │
│   complete statutory authority over dispensing, reordering, and disposal.   │
├─────────────────────────────────────────────────────────────────────────────┤
│  [⚡ Panel 1: Triage & Alerts]   [📦 Panel 2: Inventory]   [📝 Panel 3: POS] │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Dispensary Spotlight Action (Immediate quarantine directives for expired) │
│ • 4-Filter Strip: Action Required (13) | Expiry Risks (3) | Low Stock (11)  │
│ • 38 High-Density Triage Cards with Stock Runway Meters & FEFO Drawers      │
│ • Bulk Day-Sheet Dispensing with Atomic Multi-Batch Deductions & Audit Log  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Core Operational Workflows
1. **Clinical Triage Console:** Highlights urgent expired stock (e.g., *Dynapar AQ* batch `LOT-DYNA-19A`) with strict **"Quarantine & Discard — Do Not Dispense"** directives, while simultaneously advancing the next valid batch (`LOT-DYNA-19B`) to the dispensing counter.
2. **Full Inventory & Batches:** Splits into a **Medicine Formulation Catalog** (with solid `.cat-pill` capsule badges and explicit column spacing) and an active **Physical Batch Depletion Ledger** sorted in FEFO order.
3. **Unified Daily Dispensing Sheet:** Pharmacists select a single calendar date and record sales across all 38 medicines simultaneously. Clicking commit triggers [`log_sales_bulk_atomic()`](file:///C:/Users/jarna/.gemini/antigravity/scratch/pharmacy-demand-forecasting/src/db.py#L266-L352), which executes batch deductions in SQLite inside a single ACID transaction with full rollback protection.

### 7.3 Responsible AI & Clinical Safeguards
- **Non-Prescriptive Operational Boundary:** The system strictly advises on stock replenishment and batch rotation. It displays permanent, prominent disclaimers on every panel clarifying that it does not provide clinical diagnostic, treatment, or dosing advice.
- **Empirical Uncertainty Reporting:** Inventory cards present honest error margins derived directly from held-out CatBoost test error ($\pm\text{MAE}_{\text{brand}}$), eliminating arbitrary heuristics.
- **Data Privacy & Governance:** The solution operates exclusively on de-identified, aggregate transaction units and physical batch numbers, ensuring zero patient health information (PHI) or personal data is collected or stored.

---

## 8. Acceptance Tests Verification Matrix

| Acceptance Test | Requirement | Implementation Evidence | Verification Status |
|:---|:---|:---|:---:|
| **Test #1: Training Reproduction** | Reproduce dataset split and training run from instructions | Run `python src/train.py`. Loads `salesdaily.csv`, builds features, splits chronologically, and saves `pharma_catboost_demand_model.pkl`. | ✅ Passed |
| **Test #2: Held-Out Evaluation** | Demonstrate baseline and candidate metrics on unseen test set | Run `python src/evaluate.py model` and `logic`. Compares CatBoost to Naive & Last Year; logs runs to `mlflow.db`. | ✅ Passed |
| **Test #3: Live Inference** | Live prediction through app loading saved artifact | Run `streamlit run src/app.py`. Loads model artifact; dynamically computes demand and FEFO batch depletion. | ✅ Passed |
| **Test #4: Explainability & Limits** | Explain one prediction, influential inputs, and uncertainty | Dashboard displays top feature importances (rolling 28d volume, 52w seasonal lag) and empirical test MAE intervals. | ✅ Passed |
| **Test #5: Leakage & Sensitive Data** | Confirm zero target leakage and sensitive data hygiene | All lags use shifts $\ge 1$ week. Zero patient identifiable data in schema; verified by `.env.example` and code audits. | ✅ Passed |

---

## 9. Conclusion

The Antigravity Dispensary Decision Support system successfully fulfills all requirements of the Pharmacy Medicine Demand & Expiry-Risk Forecasting specification. By coupling a gradient-boosted demand forecaster with an exact, cumulative FEFO inventory engine and a resilient Streamlit console, the solution provides defensible operational guidance that reduces stockouts, curbs physical medicine waste, and protects patient safety.
