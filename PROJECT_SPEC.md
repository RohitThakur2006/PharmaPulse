# PROJECT_SPEC.md — Problem Definition & Technical Specification

## 1. Business Problem

A pharmacy owner needs to answer two questions every Monday morning:
1. **"Which medicines will I run out of before my next supplier delivery?"** → Reorder Alert
2. **"Which specific batches on my shelves will expire before customers buy them?"** → Expiry-Risk Alert

This tool provides data-driven answers to both questions. It is **advisory only** — a human pharmacist makes all final ordering and rotation decisions.

## 2. Prediction Definitions

### 2.1 Demand Forecast (Regression)

| Element | Definition |
|---------|-----------|
| **Prediction unit** | One medicine category × one prediction week |
| **Target variable** | `target_next_28d` = total units sold of that category in the upcoming 4 weeks (28 days) |
| **Prediction horizon** | 28 days ahead |
| **Point-in-time** | Every Monday, using only data known up to Sunday night |
| **Granularity** | Weekly sliding window over daily sales data |
| **Number of series** | 8 ATC medicine categories (global model trained across all 8) |

### 2.2 Expiry-Risk Flag (Binary Classification via Rule)

| Element | Definition |
|---------|-----------|
| **Prediction unit** | One batch of one medicine category |
| **Target variable** | `will_waste` = 1 if any units remain unsold at batch expiry date, else 0 |
| **Risk rule** | `remaining_batch_stock > forecasted_demand_until_expiry` → Flag = 1 (At Risk) |
| **Depletion method** | FEFO (First Expiring, First Out): demand is allocated to the earliest-expiring batch first |
| **Flag window** | Batches expiring within the next 28 days from prediction date |

### 2.3 Why the Risk Rule Is Not a Separate ML Model

The demand forecast is the stochastic (ML) component — predicting future human purchasing behaviour. Batch depletion is deterministic inventory arithmetic: comparing predicted demand against known physical stock. A separate ML classifier would:
- Risk contradicting the demand model's own forecast
- Lose the ability to output exact surplus quantities (pharmacist needs "30 boxes will expire", not "78% probability")
- Introduce double error propagation

The risk rule is forecast-driven, fully transparent, and gives actionable unit quantities. It is evaluated with Precision, Recall, F1, and Confusion Matrix against ground-truth batch outcomes.

## 3. Data Source

| Property | Value |
|----------|-------|
| **Dataset** | Pharma Sales Data (Milan Zdravković) |
| **Platform** | Kaggle |
| **License** | CC0: Public Domain |
| **Link** | https://www.kaggle.com/datasets/milkov/pharma-sales-data |
| **Origin** | Real pharmacy POS system, ~600,000 transactions over 6 years |
| **File used** | `salesdaily.csv` (2,106 rows × 13 columns) |
| **Date range** | 2014-01-02 to 2019-10-08 |
| **Pre-processing by source** | Outlier treatment and missing data imputation already applied |

### 3.1 Medicine Categories (8 ATC Codes)

| ATC Code | Category Name |
|----------|--------------|
| M01AB | Anti-inflammatory, Acetic acid derivatives |
| M01AE | Anti-inflammatory, Propionic acid derivatives |
| N02BA | Analgesics, Salicylic acid derivatives |
| N02BE | Analgesics, Paracetamol & Anilides |
| N05B | Psycholeptics, Anxiolytics |
| N05C | Psycholeptics, Hypnotics & Sedatives |
| R03 | Respiratory / Airway disease drugs |
| R06 | Antihistamines for systemic use |

### 3.2 Batch Inventory Data

The original dataset contains no batch or stock information. Batch data is **generated deterministically** from the real sales history to enable expiry-risk evaluation. The generation process is fully documented and reproducible. See `src/generate_batches.py`.

## 4. Feature Engineering

All features are computed from data known at prediction time (Sunday night). No future leakage.

### 4.1 Demand Model Features

| Feature | Description | Leakage-safe? |
|---------|-------------|:---:|
| `sales_last_1w` | Units sold in the most recent completed week | ✅ |
| `sales_last_2w` | Units sold 2 weeks ago | ✅ |
| `sales_last_4w` | Units sold 4 weeks ago | ✅ |
| `sales_same_week_last_year` | Units sold in the same week 52 weeks ago | ✅ |
| `sales_last_28d` | Sum of sales over the past 4 weeks | ✅ |
| `avg_last_90d` | Mean weekly sales over the past 12 weeks | ✅ |
| `avg_last_365d` | Mean weekly sales over the past 52 weeks | ✅ |
| `volatility_weekly_sales` | Std dev of weekly sales over the past 12 weeks | ✅ |
| `season_*` | One-hot encoded season (Winter/Spring/Summer/Fall) | ✅ |
| `category_*` | One-hot encoded medicine category | ✅ |

### 4.2 Leakage Analysis

- **Target (`target_next_28d`)** is the sum of `weekly_sales` for weeks `t+1` through `t+4`. It is never used as a feature.
- All lag and rolling features use `.shift(1)` or greater, ensuring the current week's sales are excluded.
- Train/Validation/Test splits are strictly chronological. No shuffling.
- Calendar features (month, season) are derived from the prediction date, not from future dates.

## 5. Model Comparison

### 5.1 Demand Forecasting Models

| Model | Test MAE (units) | Test RMSE | Test WAPE (%) |
|-------|:---:|:---:|:---:|
| Naive Baseline (last 28d = next 28d) | 55.18 | 114.81 | 26.91 |
| Linear Regression | 46.08 | 82.21 | 22.47 |
| Random Forest (Tuned) | 41.99 | 78.62 | 20.48 |
| XGBoost (Tuned) | 40.52 | 76.06 | 19.76 |
| LightGBM (Tuned) | 39.05 | 71.01 | 19.04 |
| **CatBoost (Tuned)** | **35.13** | **58.52** | **17.13** |

**Selected model:** CatBoost (depth=8, learning_rate=0.04, l2_leaf_reg=1.0, iterations=300).

### 5.2 Chronological Split

| Split | Period | Rows | Purpose |
|-------|--------|:---:|---------|
| Train | 2015-01 to 2018-04 | 1,376 | Learn patterns |
| Validation | 2018-04 to 2019-01 | 296 | Hyperparameter tuning |
| Test | 2019-01 to 2019-09 | 296 | Final unbiased evaluation |

### 5.3 Expiry-Risk Evaluation

| Metric | Value |
|--------|-------|
| Precision | TBD (after batch generation) |
| Recall | TBD |
| F1-Score | TBD |
| Confusion Matrix | TBD |

## 6. Evaluation Metrics

| Output | Metric | What it measures |
|--------|--------|-----------------|
| Demand | MAE | Average unit error per prediction |
| Demand | RMSE | Penalty for large forecasting disasters |
| Demand | WAPE | Store-wide percentage accuracy (handles zero-sales gracefully) |
| Expiry Risk | Precision | When we flag a batch, how often is it truly at risk? |
| Expiry Risk | Recall | Out of all batches that actually wasted, how many did we catch? |
| Expiry Risk | F1-Score | Harmonic mean of Precision and Recall |
| Expiry Risk | Confusion Matrix | Full TP/FP/TN/FN breakdown |

## 7. Inference Interface

A Streamlit web application that:
1. Loads the saved CatBoost model artifact (`.cbm` / `.pkl`).
2. Accepts current inventory data (stock levels, batch quantities, expiry dates).
3. Computes demand forecasts and runs the FEFO expiry-risk rule.
4. Displays a Monday Morning Advisory Report with:
   - Ranked reorder alerts (Urgent / Warning / Safe)
   - Batch expiry-risk flags with exact surplus quantities
   - Disclaimer: *"Advisory only. Not medical advice."*

## 8. Limitations (Must Be Stated)

1. **Only 8 aggregated ATC categories** — not individual drug brands. Real pharmacies stock hundreds of SKUs.
2. **No external demand drivers** — no weather, flu outbreak, or promotional data included.
3. **Batch data is simulated** — no real batch inventory was available in the source dataset.
4. **Single pharmacy** — trained on one store's POS data; may not generalize to different regions or pharmacy sizes.
5. **Volatile categories (R03 Respiratory, R06 Antihistamines) have higher error** — WAPE 25–37% versus 10–16% for stable categories.
6. **Model assumes stable supply** — does not account for supplier stockouts or delivery delays.
