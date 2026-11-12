# AGENTS.md — Project Rules & Agent Instructions

## Project Title
**Pharmacy Medicine Demand & Expiry-Risk Forecasting**

## What This Project Is
A decision-support ML tool for a pharmacy owner. It has two outputs:
1. **Demand Forecast:** Predict total units sold per medicine category over the next 28 days.
2. **Expiry-Risk Flags:** For each stock batch, flag whether units will remain unsold at expiry using a forecast-driven FEFO depletion rule.

**This tool is advisory only. It supports human review. It never gives medication, dosage, or treatment advice.**

## Dataset
- **Source:** "Pharma Sales Data" by Milan Zdravković (Kaggle). CC0: Public Domain license.
- **Link:** https://www.kaggle.com/datasets/milkov/pharma-sales-data
- **Contents:** 2,106 daily sales rows across 8 ATC drug categories from a real pharmacy POS system (2014–2019).
- **Batch data:** Generated deterministically from real sales patterns. Documented in `PROJECT_SPEC.md`.

## Architecture (Two Engines, One ML Model)

```
Historical Sales Data
        │
        ▼
[ CatBoost Demand Forecast Model ]  ← The only ML model
        │
        ▼  Predicted demand for next 28 days
        │
┌───────┴──────────────────────────────────────┐
│           CORE BUSINESS LOGIC                │
│                                              │
│  A. Reorder Alert    B. Expiry-Risk Flag     │
│  (Stock vs Forecast) (Batch FEFO Depletion)  │
└──────────────────────────────────────────────┘
        │
        ▼
  Monday Morning Report (Advisory)
```

## Hard Rules — NEVER Do These

### Data Integrity
- **NEVER use future data as a feature.** All features must be computable from data known at prediction time (Sunday night).
- **NEVER use random train/test splits.** All splits are strictly chronological.
- **NEVER include the target variable or any derivative of it as a feature.**
- **NEVER hard-code predictions.** The inference API must load the saved model artifact and compute predictions live.

### Scope
- **NEVER present output as medical or treatment advice.** All outputs are operational inventory guidance only.
- **NEVER commit .env files, passwords, API keys, or personal data.**
- **NEVER upload restricted or private patient/customer data.**

### Code Quality
- **NEVER add features, abstractions, or error handling that wasn't requested.**
- **NEVER refactor adjacent code that isn't broken.**
- **NEVER silently pick between ambiguous interpretations — surface them and ask.**

## Agent Behaviour Rules

### 1. Think Before Coding
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.

### 2. Simplicity First
- Minimum code that solves the problem. Nothing speculative.
- No abstractions for single-use code.
- If you write 200 lines and it could be 50, rewrite it.

### 3. Surgical Changes
- Touch only what you must. Clean up only your own mess.
- Match existing style, even if you'd do it differently.
- Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution
- Define success criteria. Loop until verified.
- For multi-step tasks, state a brief plan with verification checks.

## Key Commands

```bash
# Environment setup
pip install -r requirements.txt

# Training pipeline (reproduces everything from raw data)
python src/train.py

# Run inference API
python src/app.py

# Run tests
pytest tests/
```

## Random Seeds
- All models use `random_seed=42` (CatBoost) or `random_state=42` (sklearn/LightGBM/XGBoost).
- NumPy seed: `np.random.seed(42)` where applicable.

## Metrics to Report
| Output | Metrics |
|--------|---------|
| Demand Forecast (Regression) | MAE, RMSE, WAPE |
| Expiry-Risk Flags (Binary) | Precision, Recall, F1-Score, Confusion Matrix |
