# Pharmacy Medicine Demand & Expiry-Risk Decision Support

[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](https://www.python.org/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Tests: 4 Passing](https://img.shields.io/badge/tests-4%20passed-brightgreen.svg)](tests/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-3.17.2-0194E2.svg)](mlflow.db)

A clinical decision-support machine learning solution for community pharmacies that couples:
1. **28-Day Demand Forecasting:** A tuned global CatBoost regressor predicting 4-week demand across 8 anatomical therapeutic chemical (ATC) classes and 38 Indian household medicine brands.
2. **Cumulative FEFO Expiry-Risk Engine:** Inventory depletion logic that flags batches at risk of expiring unsold on the shelf with exact surplus quantities.
3. **Clinical Governance Dashboard:** A dark-mode Streamlit dispensary console with real-time atomic point-of-sale dispensing, bulk day sheets, and non-prescriptive advisory safeguards.

---

## Architecture Overview

```
                      ┌─────────────────────────────────────────┐
                      │  6-Year Historical Sales (Kaggle CC0)   │
                      │  2,106 weekly records × 8 ATC categories │
                      └───────────────────┬─────────────────────┘
                                          │
                                          ▼
                      ┌─────────────────────────────────────────┐
                      │    features.py: Lags, 28d/90d/365d      │
                      │    Rolling Sums, Met Season Dummies     │
                      └───────────────────┬─────────────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
             ┌─────────────────────┐             ┌─────────────────────┐
             │   Tier 1: Demand    │             │   Tier 2: Logic     │
             │   CatBoost Regressor│             │   FEFO Depletion    │
             │   WAPE: 17.18%      │             │   F1-Score: 0.850   │
             │   MAE: 37.21 units  │             │   Precision: 0.835  │
             └──────────┬──────────┘             │   Recall: 0.864     │
                        │                        └──────────┬──────────┘
                        └─────────────────┬─────────────────┘
                                          │
                                          ▼
                      ┌─────────────────────────────────────────┐
                      │        app.py (Streamlit SaaS UI)       │
                      │  • Panel 1: Clinical Triage & Alerts    │
                      │  • Panel 2: Full Inventory & Batches    │
                      │  • Panel 3: POS Bulk Dispense Sheet     │
                      └─────────────────────────────────────────┘
```

---

## 1. Quickstart & Installation

### Prerequisites
- Python 3.10 to 3.14
- Git

### Installation
```powershell
# Clone the repository
git clone https://github.com/rohit6thakur9/pharmacy-demand-forecasting.git
cd pharmacy-demand-forecasting

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Reproducing the Pipeline (Acceptance Tests)

### Acceptance Test #1: Reproduce Training Pipeline
Trains the CatBoost model on chronological splits (Train 2014–2017, Val 2018, Test 2019), computes baseline comparisons, and exports the production model artifact:
```powershell
python src/train.py
```
*Outputs: `pharma_catboost_demand_model.pkl` and `model_metadata.json`.*

---

### Acceptance Test #2: Defensible Evaluation Harness & MLflow
Run the two-tier evaluation harness:
```powershell
# Tier 1: Forecast accuracy on 52 rolling origins vs Naive & Last-Year baselines
python src/evaluate.py model

# Tier 2: End-to-end batch expiry logic on 3,456 single-batch scenarios + 288 FEFO groups
python src/evaluate.py logic
```

To launch the MLflow tracking dashboard and inspect logged parameters, metrics, and CSV artifacts:
```powershell
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5001
```
Open your browser to: **`http://localhost:5001`**

---

### Acceptance Test #3: Interactive Dispensary Inference App
Launch the interactive Streamlit decision-support application:
```powershell
streamlit run src/app.py
```
Features demonstrated live:
- **Zero hard-coded sample predictions:** Every demand number is computed dynamically from CatBoost inference over the SQLite sales history.
- **Dynamic FEFO queue peeking:** Selecting any medicine displays its active front lot and days to expiry.
- **Bulk Day-Sheet Dispensing:** Submit a complete day's patient sales in a single atomic transaction.

---

### Acceptance Test #4: Prediction Explainability & Uncertainty
- **Feature Importance:** Top drivers (4-week trailing demand `sales_last_28d`, 52-week lag `sales_same_week_last_year`, and seasonality) are tracked in `model_metadata.json` and visible in the app.
- **Empirical Uncertainty Bounds:** The displayed range is statistically grounded in the model's test error:
  $$\text{Expected Range} = \hat{Y} \pm (\text{MODEL\_MAE} \times p_{\text{share}})$$
  *(Eliminating arbitrary $\pm 10\%$ heuristics).*

---

### Acceptance Test #5: Data Leakage & Sensitive Data Hygiene
- **Strict Cutoff Boundary:** Lags and rolling features are shifted by $\ge 1$ week, preventing target leakage.
- **Privacy Hygiene:** No patient identifiable data (PII), proprietary vendor margins, or live credentials exist in this repository.

---

## 3. Automated Test Suite

Run the full automated test suite verifying decision engine logic and multi-medicine atomic transactions:
```powershell
pytest tests/ -v
```
All **4 test suites** pass ($4/4$ passing):
- `tests/test_engine.py`: Optimal stock, critical stockouts, and cumulative FEFO multi-batch allocation.
- `tests/test_db_bulk.py`: Multi-medicine bulk day-sheet dispensing, sequential batch exhaustion, and rollback safety.

---

## 4. Key Performance Results

| Tier | Task | Candidate Model | Baseline 1 (Naive 28d) | Baseline 2 (Last Year) | Benchmark Improvement |
|:---|:---|:---:|:---:|:---:|:---:|
| **Tier 1** | 28-day Category Demand | **WAPE: 17.18%**<br>MAE: 37.21 units | WAPE: 19.02%<br>MAE: 41.19 units | WAPE: 25.32%<br>MAE: 54.84 units | **+1.84% vs Naive**<br>**+8.14% vs Last Year** |

| Tier | Task | Precision | Recall | F1-Score | Accuracy | Base Rate |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Tier 2** | Batch Expiry Classification | **0.835** (83.5%) | **0.864** (86.4%) | **0.850** (85.0%) | **0.827** (82.7%) | 0.564 (56.4%) |

---

## 5. Repository Structure

```
pharmacy-demand-forecasting/
├── data/
│   └── salesdaily.csv              # Real pharmacy POS sales history (Milan Zdravkovic, CC0)
├── reports/                        # Exported CSV evaluation benchmarks
│   ├── tier1_per_category.csv
│   ├── tier2_confusion.csv
│   ├── tier2_breakdown.csv
│   └── tier2_scenarios.csv
├── src/
│   ├── app.py                      # Production Streamlit clinical decision-support console
│   ├── build_test_data.py          # Synthetic model history & batch scenario generator
│   ├── db.py                       # SQLite relational schema, atomic transactions, audit ledger
│   ├── engine.py                   # Pure decision engine (FEFO allocation & reorder rules)
│   ├── evaluate.py                 # Two-tier evaluation harness with MLflow logging
│   ├── features.py                 # Leakage-free feature engineering (train/serve parity)
│   ├── seed_db.py                  # Formulary seeder (38 Indian brands, 45 physical batches)
│   └── train.py                    # Standalone reproducible CatBoost training script
├── tests/
│   ├── test_engine.py              # Unit tests for core decision engine
│   └── test_db_bulk.py             # Unit tests for atomic bulk dispensing
├── .env.example                    # Environment variable configuration template
├── .gitignore                      # Git exclusion rules
├── mlflow.db                       # MLflow SQLite tracking store
├── model_metadata.json             # CatBoost hyperparameters and feature importances
├── pharma_catboost_demand_model.pkl# Trained production CatBoost model artifact
├── PROJECT_SPEC.md                 # Formal problem specification and design choices
├── README.md                       # Quickstart, architecture, and reproduction guide
├── REPORT.md                       # Comprehensive 6-section technical report
└── requirements.txt                # Pinned production dependencies
```

---

## 6. Clinical Decision Support Advisory Notice

> [!IMPORTANT]
> **Advisory Scope Only:** All demand projections, stockout alerts, suggested reorder quantities, and batch expiry assessments produced by this system are **operational decision-support aids** intended exclusively for inventory replenishment and inventory rotation. A registered, licensed pharmacist retains complete statutory authority over all purchasing, dispensing, discounting, and stock disposal decisions. This software provides zero diagnostic, therapeutic, clinical, or medication dosing advice.
