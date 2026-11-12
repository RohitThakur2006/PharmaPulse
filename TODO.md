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
- [ ] Batch history generation with ground truth labels (`src/generate_batches.py`)
- [ ] Evaluate batch flags: Confusion Matrix, Precision, Recall, F1-Score

## Phase 4: Inference Interface (Streamlit App)
- [x] Build Streamlit app (`src/app.py`)
- [x] Load saved model artifact — live inference ready
- [x] Accept inventory input (log daily sales + receive delivery forms)
- [x] Display category-level demand forecasts
- [x] Display independent Reorder & Expiry badges with color + emoji + text
- [x] Display FEFO batch risk breakdown with surplus waste quantities
- [x] Permanent advisory disclaimer banner on every screen
- [x] Model uncertainty intervals (±MAE) displayed on dashboard
- [x] Feature explainability section (top drivers) for Acceptance Test #4

## Phase 5: Reproducibility, Tests & Documentation
- [x] Create `requirements.txt` with dependencies
- [x] Create `.gitignore` (ignore `pharmacy.db`, caches, checkpoints)
- [x] Unit tests for decision engine (`tests/test_engine.py`)
- [ ] Create `README.md` with setup, data prep, training, and inference commands
- [ ] Create `.env.example` (placeholder, no real credentials)
- [ ] Write technical report (EDA, target, features, baseline, comparison, split, metrics, error analysis, limitations)
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
| Problem framing, data sourcing & EDA (15) | 🟡 Partial | Target/horizon defined, dataset sourced, EDA done in notebooks — needs consolidation |
| Baseline & model development (25) | ✅ Done | 6 models compared, hyperparameter tuning, feature engineering |
| Validation & metrics (20) | 🟡 Partial | Demand metrics done — expiry Precision/Recall/F1 pending |
| Inference interface (15) | ⏳ Not started | Streamlit app planned |
| Explainability & responsible use (10) | 🟡 Partial | Feature importance done, limitations listed — needs bias/privacy discussion |
| Reproducibility, tests & documentation (15) | ⏳ Not started | Needs scripts, tests, README, requirements.txt |
