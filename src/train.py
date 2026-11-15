"""
train.py — Standalone reproducible training script for the CatBoost demand model.
Acceptance Test #1: Reproduce the dataset split and training run.

Loads data/salesdaily.csv, aggregates daily to weekly bins, computes target_next_28d,
builds lag and rolling features via features.py, performs chronological train/val/test split,
evaluates CatBoost against Naive baselines, and saves the final artifact:
- pharma_catboost_demand_model.pkl
- model_metadata.json
"""

import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from features import (
    CATEGORY_MAPPING,
    ALL_CATEGORIES,
    FEATURE_COLUMNS,
    build_features_for_category,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "salesdaily.csv"
MODEL_PATH = ROOT / "pharma_catboost_demand_model.pkl"
META_PATH = ROOT / "model_metadata.json"

TRAIN_END = "2017-12-31"
VAL_END = "2018-12-31"


def load_and_prepare_dataset(csv_path: Path) -> pd.DataFrame:
    """Loads raw salesdaily.csv and creates weekly long-format sales series."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    raw = pd.read_csv(csv_path)
    raw["datum"] = pd.to_datetime(raw["datum"])

    # Melt to long format: ['datum', 'atc_code', 'units_sold']
    atc_cols = list(CATEGORY_MAPPING.keys())
    long_df = raw.melt(
        id_vars=["datum"],
        value_vars=atc_cols,
        var_name="atc_code",
        value_name="units_sold"
    )
    long_df["category"] = long_df["atc_code"].map(CATEGORY_MAPPING)

    # Weekly aggregation (Monday start)
    weekly = (
        long_df.groupby(["category", pd.Grouper(key="datum", freq="W-MON")])["units_sold"]
        .sum()
        .reset_index()
        .rename(columns={"datum": "week_start", "units_sold": "weekly_sales"})
    )

    all_cat_frames = []
    for cat in ALL_CATEGORIES:
        cat_df = weekly[weekly["category"] == cat].copy()
        featured = build_features_for_category(cat_df)

        # Compute target: sum of units in the upcoming 4 weeks
        # shift(-1) + shift(-2) + shift(-3) + shift(-4)
        future = (
            featured["weekly_sales"].shift(-1)
            + featured["weekly_sales"].shift(-2)
            + featured["weekly_sales"].shift(-3)
            + featured["weekly_sales"].shift(-4)
        )
        featured["target_next_28d"] = future
        featured["category"] = cat
        all_cat_frames.append(featured)

    dataset = pd.concat(all_cat_frames, axis=0).reset_index(drop=True)

    # One-hot encode seasons and categories to match features.py schema
    for season in ["Fall", "Spring", "Summer", "Winter"]:
        dataset[f"season_{season}"] = (dataset["season"] == season).astype(int)

    for cat in ALL_CATEGORIES:
        if cat != "M01AB_Anti_Inflammatory_Acetic":
            dataset[f"category_{cat}"] = (dataset["category"] == cat).astype(int)

    # Drop rows where lags or forward target cannot be computed
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["target_next_28d"]).copy()
    dataset = dataset.sort_values("week_start").reset_index(drop=True)

    logger.info(f"Prepared dataset with {len(dataset)} weekly observations across 8 categories.")
    return dataset


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    denom = float(np.sum(y_true))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "wape": float(100.0 * np.sum(np.abs(err)) / denom) if denom else 0.0,
        "bias": float(100.0 * np.sum(err) / denom) if denom else 0.0,
    }


def main():
    logger.info("Starting pharmacy demand model training pipeline...")
    df = load_and_prepare_dataset(DATA_PATH)

    # Chronological Split
    train_mask = df["week_start"] <= TRAIN_END
    val_mask = (df["week_start"] > TRAIN_END) & (df["week_start"] <= VAL_END)
    test_mask = df["week_start"] > VAL_END

    X_train = df.loc[train_mask, FEATURE_COLUMNS]
    y_train = df.loc[train_mask, "target_next_28d"]

    X_val = df.loc[val_mask, FEATURE_COLUMNS]
    y_val = df.loc[val_mask, "target_next_28d"]

    X_test = df.loc[test_mask, FEATURE_COLUMNS]
    y_test = df.loc[test_mask, "target_next_28d"]

    logger.info(f"Split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Naive Baseline Evaluation on Test Set: next 28d = last 28d
    naive_test_pred = df.loc[test_mask, "sales_last_28d"].to_numpy()
    naive_metrics = calculate_metrics(y_test.to_numpy(), naive_test_pred)
    logger.info(
        f"Naive Baseline (Test): WAPE = {naive_metrics['wape']:.2f}%, MAE = {naive_metrics['mae']:.2f}"
    )

    # Train Tuned CatBoost Regressor
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=100,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    # Evaluate on held-out 2019 Test Set
    test_pred = model.predict(X_test)
    test_pred = np.maximum(0.0, test_pred)
    test_metrics = calculate_metrics(y_test.to_numpy(), test_pred)

    logger.info("=" * 60)
    logger.info("HELD-OUT TEST RESULTS (2019 UNSEEN DATA)")
    logger.info("=" * 60)
    logger.info(f"  CatBoost Test WAPE : {test_metrics['wape']:.2f}% (vs Naive: {naive_metrics['wape']:.2f}%)")
    logger.info(f"  CatBoost Test MAE  : {test_metrics['mae']:.2f} units (vs Naive: {naive_metrics['mae']:.2f})")
    logger.info(f"  CatBoost Test RMSE : {test_metrics['rmse']:.2f} units")
    logger.info(f"  CatBoost Test Bias : {test_metrics['bias']:+.2f}%")
    logger.info("=" * 60)

    # Retrain on full Train+Val and export production artifact
    logger.info("Retraining on combined Train + Validation history for production deployment...")
    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)

    final_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
    )
    final_model.fit(X_full, y_full)

    joblib.dump(final_model, MODEL_PATH)
    logger.info(f"Exported model artifact to: {MODEL_PATH}")

    # Feature Importance Export
    feat_imp = pd.Series(final_model.get_feature_importance(), index=FEATURE_COLUMNS).sort_values(
        ascending=False
    )
    metadata = {
        "model_type": "CatBoostRegressor",
        "target": "target_next_28d (units sold next 4 weeks)",
        "prediction_horizon_days": 28,
        "features": FEATURE_COLUMNS,
        "feature_importances": feat_imp.to_dict(),
        "test_wape": round(test_metrics["wape"], 2),
        "test_mae": round(test_metrics["mae"], 2),
        "test_rmse": round(test_metrics["rmse"], 2),
        "test_bias": round(test_metrics["bias"], 2),
        "naive_test_wape": round(naive_metrics["wape"], 2),
    }

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Exported metadata to: {META_PATH}")
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
