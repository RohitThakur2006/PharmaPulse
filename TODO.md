# TODO.md — Progress Tracker

**Project:** Pharmacy Medicine Demand & Expiry-Risk Forecasting  
**Status:** In Progress  
**Last updated:** 2026-11-12

---

## Phase 0: Research & Planning
- [x] Literature review: pharmacy demand forecasting projects, papers, datasets
- [x] Dataset inspection: Zdravkovic Pharma Sales, Neuralthon Inventory, Perishable Goods
- [x] Identify data gap: no public dataset combines sales + batches + expiry
- [x] Lock decisions: target (D1), risk method (D2), data strategy (D3), granularity (D4)
- [x] Create project documentation (AGENTS.md, PROJECT_SPEC.md, DECISIONS.md, TODO.md)

## Phase 1: Data Preparation & EDA
- [x] Load `salesdaily.csv`, rename ATC codes to readable names
- [x] Reshape wide → long format (8 categories × 2,106 days = 16,848 rows)
- [x] Aggregate daily → weekly (Monday-to-Sunday bins)
- [x] Compute target: exact sum of next 4 weeks (`shift(-1) + shift(-2) + shift(-3) + shift(-4)`)
- [x] Engineer lag features (1w, 2w, 4w, 52w)
- [x] Engineer rolling features (28d sum, 90d avg, 365d avg, 12w volatility)
- [x] Add season dummy variables (Winter/Spring/Summer/Fall)
- [x] Drop `ratio_28d_to_90d` (noise, 0.08 correlation)
- [x] Correlation analysis and heatmap
- [x] Verify target alignment (manual arithmetic check on row 0)
- [ ] Consolidate EDA into a clean reproducible notebook with visualizations
- [ ] Leakage analysis documentation (write up in technical report)
- [ ] Data quality checks documentation

## Phase 2: Baseline & Model Development
- [x] Naive baseline (last 28 days = next 28 days): WAPE 26.91%
- [x] Linear Regression: WAPE 22.47%
- [x] Random Forest: WAPE 20.48%
- [x] XGBoost: WAPE 19.76%
- [x] LightGBM: WAPE 19.04%
- [x] CatBoost: WAPE 18.55% (default) → 17.13% (tuned)
- [x] Hyperparameter grid search on validation set for all 4 tree models
- [x] Final model: retrain CatBoost on Train+Val, evaluate on unseen 2019 Test
- [x] Monthly vs Weekly experiment (proved weekly is 10+ WAPE points better)
- [x] Feature importance analysis (CatBoost)
- [x] Category-level drill-down (per-medicine WAPE breakdown)
- [x] Save model artifact (.cbm, .pkl) and metadata (.json)
- [ ] Consolidate training pipeline into `src/train.py` (reproducible script)
- [ ] Actual vs Predicted visualization (corrected alignment)

## Phase 3: Batch Generation & Expiry-Risk Evaluation
- [x] Design batch generation & FEFO depletion logic in `src/engine.py`
- [x] Unit test cumulative FEFO allocation against core scenarios (`tests/test_engine.py`)
- [x] Implement database schema with atomic transactions and audit ledger (`src/db.py`)
- [x] Implement realistic sample inventory seeding (`src/seed_db.py`)
- [x] Batch history generation with ground truth labels (`pharmacy_test_logic.db`, `build_test_data.py`)
- [x] Synthetic test history generation (`pharmacy_test_model.db`, `build_test_data.py`)
- [x] Tier 1 forecast evaluation vs Naive & Last-Year baselines across 52 rolling origins (`src/evaluate.py model`)
- [x] Tier 2 expiry risk evaluation with Confusion Matrix, Precision (0.835), Recall (0.864), F1 (0.850) (`src/evaluate.py logic`)
- [x] MLflow experiment tracking integration (`mlflow.db`, `mlruns/`, CSV artifacts in `reports/`)

## Phase 4: Inference Interface (Streamlit App)
- [x] Build Streamlit app (`src/app.py`) with SaaS Forest Green & Electric Purple theme
- [x] Load saved model artifact — live inference across 8 ATC categories
- [x] 38 Essential Indian Household Medicines mapped to formulary with brand-share demand scaling
- [x] Display category-level & brand-level demand forecasts with empirical uncertainty bounds (`±MODEL_MAE × brand_share`)
- [x] Three prominent top navigation venues:
  - Panel 1: Clinical Triage & Alerts (Spotlight Action card, 4-filter triage strip, stock runway meters, FEFO batch drawers)
  - Panel 2: Full Inventory & Batches (Venue 1: 38-medicine formulation catalog, Venue 2: Physical batch depletion ledger)
  - Panel 3: Dispensing Ledger (Operation 1: Comprehensive day-end bulk dispensing sheet, Operation 2: Receive new batch, Operation 3: Register brand-new medicine)
- [x] Real-time atomic FEFO stock deduction and audit logging
- [x] Permanent clinical decision-support advisory notice on all panels

## Phase 5: Reproducibility, Tests & Documentation
- [x] Create `requirements.txt` with dependencies (including `mlflow`, `joblib`, `catboost`)
- [x] Create `.gitignore` (ignore `pharmacy.db`, caches, checkpoints, test DBs)
- [x] Unit tests for decision engine (`tests/test_engine.py`)
- [x] Unit tests for atomic bulk dispensing (`tests/test_db_bulk.py`)
- [ ] Consolidate training pipeline into `src/train.py` (reproducible training script)
- [ ] Create `README.md` with setup, evaluation, and dashboard instructions
- [ ] Create `.env.example` (environment configuration template)
- [ ] Write comprehensive technical report (`REPORT.md`)
- [ ] Record 3–5 minute demo video

## Phase 6: Submission
- [ ] Final pipeline re-run
- [ ] Verify all links work (repo, report, video)
- [ ] Confirm no .env / passwords / personal data committed
- [ ] Fill submission form (all required fields)

---

## Rubric Coverage Checklist

| Criterion (Marks) | Status | Evidence |
|:---|:---:|:---|
| Problem framing, data sourcing & EDA (15) | 🟡 Partial | Target/horizon defined, 8 ATC categories, EDA done — report writeup pending |
| Baseline & model development (25) | ✅ Done | 6 models compared, tuned CatBoost (WAPE 17.18%), feature importance, rolling origins |
| Validation & metrics (20) | ✅ Done | MLflow tracking (`mlflow.db`), Tier 1 demand metrics vs 2 baselines, Tier 2 F1 0.850 / Precision 0.835 / Recall 0.864 |
| Inference interface (15) | ✅ Done | Full Streamlit decision-support dashboard (`app.py`), 3 panels, 38 Indian brands, bulk POS sheet |
| Explainability & responsible use (10) | ✅ Done | Feature drivers, clinical advisory disclaimers, honest ±MAE uncertainty bounds, error analysis (0.5×/0.9× phenomena) |
| Reproducibility, tests & documentation (15) | 🟡 Partial | 4/4 pytest suites passing, requirements.txt, .gitignore done — README.md & train.py pending |
