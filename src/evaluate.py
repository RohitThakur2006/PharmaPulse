"""
evaluate.py — Evaluation harness for the pharmacy decision-support tool.

Tier 1  (`model`)  Forecast accuracy. The CatBoost artifact is scored on the
                   synthetic history in pharmacy_test_model.db: 28-day forecasts
                   every 28 days, compared with what actually sold, per category,
                   alongside a naive and a same-period-last-year baseline.

Tier 2  (`logic`)  Expiry flags. The real sales history plus the generated batch
                   grid in pharmacy_test_logic.db: precision, recall, F1 and a
                   confusion matrix for the at-risk flag, split by expiry window
                   and quantity level, plus a FEFO ordering check.

    python src/evaluate.py model
    python src/evaluate.py logic

Each tier is one MLflow run under the "pharma-eval" experiment. Run metadata
lives in ./mlflow.db (SQLite — MLflow 3 no longer accepts a plain directory),
logged artifacts under ./mlruns. The harness imports features.py and engine.py —
it never reimplements them, so it always measures the code the app runs.
"""

import hashlib
import sqlite3
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from engine import evaluate_batch_expiry_fefo
from features import CATEGORY_MAPPING, FEATURE_COLUMNS, prepare_inference_row

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "pharma_catboost_demand_model.pkl"
MODEL_DB = ROOT / "pharmacy_test_model.db"
LOGIC_DB = ROOT / "pharmacy_test_logic.db"
REPORTS = ROOT / "reports"
TRACKING = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"

EXPERIMENT = "pharma-eval"
HORIZON = 28
STEP = 28
ORIGIN_START = "2016-01-01"
ORIGIN_END = "2019-12-03"

ONE_DAY = pd.Timedelta(days=1)
HORIZON_TD = pd.Timedelta(days=HORIZON)
WASTE_FLAGS = {"CRITICAL_WASTE", "MODERATE_WASTE"}

SHORT = {v: k for k, v in CATEGORY_MAPPING.items()}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def load_model():
    """Loads the saved artifact. Predictions are always computed live."""
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model artifact not found: {MODEL_PATH.name}")
    return joblib.load(MODEL_PATH)


def read_sales(db_path: Path) -> pd.DataFrame:
    """Daily units per ATC category: brands summed back up to their category."""
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path.name}\n"
                         "Run: python src/build_test_data.py")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT s.sale_date AS datum, p.category_full_name AS category, s.units_sold "
        "FROM daily_sales s JOIN products p ON s.product_id = p.product_id",
        conn,
    )
    conn.close()
    df["datum"] = pd.to_datetime(df["datum"])
    return df.groupby(["datum", "category"], as_index=False)["units_sold"].sum()


def to_wide(sales: pd.DataFrame) -> pd.DataFrame:
    return sales.pivot_table(index="datum", columns="category",
                             values="units_sold", aggfunc="sum").sort_index()


def metrics(actual: pd.Series, pred: pd.Series) -> dict:
    """MAE, RMSE, WAPE and bias. Bias is signed: positive means over-forecast."""
    err = pred.to_numpy() - actual.to_numpy()
    denom = float(actual.sum())
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "wape": 100 * float(np.abs(err).sum()) / denom if denom else float("nan"),
        "bias": 100 * float(err.sum()) / denom if denom else float("nan"),
    }


def confusion(df: pd.DataFrame) -> dict:
    """Flag quality against the truth labels already sitting in `df`."""
    pred, truth = df["predicted"], df["actual"]
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else float("nan"))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(df) if len(df) else float("nan"),
        "base_rate": float(truth.mean()),
    }


def clean(values: dict) -> dict:
    """Drops NaN so MLflow never receives an unusable metric."""
    return {k: float(v) for k, v in values.items() if np.isfinite(v)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def common_params(dataset: str) -> dict:
    return {
        "dataset": dataset,
        "model": MODEL_PATH.name,
        "model_sha256": digest(MODEL_PATH),
        "target": "next 28 days total units",
        "horizon_days": HORIZON,
        "git_commit": git_commit(),
    }


# ---------------------------------------------------------------------------
# Tier 1 — forecast accuracy
# ---------------------------------------------------------------------------
def evaluate_model() -> None:
    model = load_model()
    sales = read_sales(MODEL_DB)
    wide = to_wide(sales)
    origins = pd.date_range(ORIGIN_START, ORIGIN_END, freq=f"{STEP}D")

    rows, skipped = [], []
    for as_of in origins:
        features = prepare_inference_row(sales, as_of)
        if features[FEATURE_COLUMNS].isna().any().any():
            skipped.append(as_of)
            continue
        preds = model.predict(features[FEATURE_COLUMNS])

        forward = wide.loc[as_of + ONE_DAY: as_of + HORIZON_TD]
        trailing = wide.loc[as_of - pd.Timedelta(days=27): as_of]
        last_year = wide.loc[as_of - pd.Timedelta(days=364):
                              as_of - pd.Timedelta(days=337)]

        for cat, pred in zip(features["category"], preds):
            rows.append({
                "origin": as_of,
                "category": cat,
                "actual": float(forward[cat].sum()),
                "pred": max(0.0, float(pred)),
                "naive": float(trailing[cat].sum()),
                "last_year": float(last_year[cat].sum()),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No scorable forecast origins — is the history long enough?")

    overall = {
        "model": metrics(df["actual"], df["pred"]),
        "naive": metrics(df["actual"], df["naive"]),
        "last_year": metrics(df["actual"], df["last_year"]),
    }

    per_cat = []
    for cat, g in df.groupby("category"):
        m = metrics(g["actual"], g["pred"])
        n = metrics(g["actual"], g["naive"])
        ly = metrics(g["actual"], g["last_year"])
        per_cat.append({
            "category": SHORT[cat],
            "n": len(g),
            "actual_units": int(round(g["actual"].sum())),
            "mae": round(m["mae"], 2),
            "rmse": round(m["rmse"], 2),
            "wape": round(m["wape"], 2),
            "bias": round(m["bias"], 2),
            "naive_wape": round(n["wape"], 2),
            "last_year_wape": round(ly["wape"], 2),
            "skill_vs_naive": round(n["wape"] - m["wape"], 2),
        })
    per_cat = pd.DataFrame(per_cat).sort_values("actual_units", ascending=False)
    per_cat = pd.concat([pd.DataFrame([{
        "category": "ALL",
        "n": len(df),
        "actual_units": int(round(df["actual"].sum())),
        "mae": round(overall["model"]["mae"], 2),
        "rmse": round(overall["model"]["rmse"], 2),
        "wape": round(overall["model"]["wape"], 2),
        "bias": round(overall["model"]["bias"], 2),
        "naive_wape": round(overall["naive"]["wape"], 2),
        "last_year_wape": round(overall["last_year"]["wape"], 2),
        "skill_vs_naive": round(overall["naive"]["wape"]
                                - overall["model"]["wape"], 2),
    }]), per_cat], ignore_index=True)

    # ---- report ----
    print("=" * 78)
    print("TIER 1  forecast accuracy on the synthetic history")
    print("=" * 78)
    print(f"  origins           : {len(origins)} every {STEP} days "
          f"({origins[0].date()} -> {origins[-1].date()})")
    if skipped:
        print(f"  skipped           : {len(skipped)} (features incomplete, "
              f"first {skipped[0].date()})")
    print(f"  forecasts scored  : {len(df):,} "
          f"({len(df) // 8} origins x {df['category'].nunique()} categories)")
    print(f"  horizon           : {HORIZON} days")
    print(f"  units scored      : {int(df['actual'].sum()):,}")
    print()
    print(f"  {'forecast':10s} {'MAE':>9s} {'RMSE':>9s} {'WAPE%':>8s} {'bias%':>8s}")
    for name, key in [("model", "model"), ("naive", "naive"),
                      ("last year", "last_year")]:
        m = overall[key]
        print(f"  {name:10s} {m['mae']:9.2f} {m['rmse']:9.2f} "
              f"{m['wape']:8.2f} {m['bias']:+8.2f}")
    print()
    print(per_cat.to_string(index=False))
    print()
    print(f"  The model beats 'next 28 days = last 28 days' by "
          f"{overall['naive']['wape'] - overall['model']['wape']:.2f} WAPE points "
          f"and the same-period-last-year baseline by "
          f"{overall['last_year']['wape'] - overall['model']['wape']:.2f} points.")

    # ---- artifacts + MLflow ----
    REPORTS.mkdir(exist_ok=True)
    csv_path = REPORTS / "tier1_per_category.csv"
    per_cat.to_csv(csv_path, index=False)

    with mlflow.start_run(run_name="tier1-model"):
        mlflow.log_params({
            **common_params(MODEL_DB.name),
            "split": "chronological, rolling origin",
            "origins": len(df) // 8,
            "origin_step_days": STEP,
            "forecasts": len(df),
        })
        mlflow.log_metrics(clean({
            "mae": overall["model"]["mae"],
            "rmse": overall["model"]["rmse"],
            "wape_pct": overall["model"]["wape"],
            "bias_pct": overall["model"]["bias"],
            "naive_wape_pct": overall["naive"]["wape"],
            "last_year_wape_pct": overall["last_year"]["wape"],
            "skill_vs_naive_pts": overall["naive"]["wape"] - overall["model"]["wape"],
            "skill_vs_last_year_pts": (overall["last_year"]["wape"]
                                       - overall["model"]["wape"]),
        }))
        for _, r in per_cat[per_cat["category"] != "ALL"].iterrows():
            mlflow.log_metrics(clean({
                f"wape_{r['category']}": float(r["wape"]),
                f"mae_{r['category']}": float(r["mae"]),
            }))
        mlflow.log_artifact(str(csv_path))
        print(f"\n  MLflow run logged. Report: {csv_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Tier 2 — expiry flags
# ---------------------------------------------------------------------------
def evaluate_logic() -> None:
    model = load_model()
    sales = read_sales(LOGIC_DB)
    wide = to_wide(sales)

    conn = sqlite3.connect(LOGIC_DB)
    batches = pd.read_sql_query("SELECT * FROM test_batches", conn)
    conn.close()
    batches["as_of"] = pd.to_datetime(batches["as_of_date"])

    # One live forecast per as-of date and category.
    forecasts = {}
    for as_of in sorted(batches["as_of"].unique()):
        features = prepare_inference_row(sales, pd.Timestamp(as_of))
        if features[FEATURE_COLUMNS].isna().any().any():
            raise SystemExit(f"Features incomplete at {as_of}")
        preds = model.predict(features[FEATURE_COLUMNS])
        for cat, pred in zip(features["category"], preds):
            # features.py returns the full category name; test_batches stores the
            # short ATC code, so key the map on the short code.
            forecasts[(pd.Timestamp(as_of), SHORT[cat])] = max(0.0, float(pred))

    # ---- single-batch scenarios ----
    core = batches[batches["batch_id"].str.startswith("T2-G")]
    scored = []
    for b in core.itertuples(index=False):
        forecast = forecasts[(b.as_of, b.atc_code)]
        out = evaluate_batch_expiry_fefo(
            [{"batch_id": b.batch_id,
              "quantity_on_hand": b.quantity_on_hand,
              "expiry_date": b.expiry_date}],
            forecast, b.as_of_date, HORIZON,
        )[0]
        waste = b.quantity_on_hand - b.actual_units_window
        scored.append({
            "batch_id": b.batch_id,
            "category": b.atc_code,
            "expiry_window_days": b.expiry_window_days,
            "qty_multiplier": b.qty_multiplier,
            "flag": out["expiry_flag"],
            "predicted_waste": out["expected_waste_units"],
            "true_waste": max(0.0, waste),
            "predicted": out["expiry_flag"] in WASTE_FLAGS,
            "actual": waste > 0,
        })
    scored = pd.DataFrame(scored)
    m = confusion(scored)

    # ---- FEFO groups ----
    fefo = batches[batches["batch_id"].str.startswith("T2-F")]
    groups = ordered = identity = agreed = 0
    for _, g in fefo.groupby("scenario_id"):
        as_of, cat = g["as_of"].iloc[0], g["atc_code"].iloc[0]
        forecast = forecasts[(as_of, cat)]
        rows_in = [{"batch_id": r.batch_id,
                    "quantity_on_hand": r.quantity_on_hand,
                    "expiry_date": r.expiry_date} for r in g.itertuples(index=False)]
        outs = evaluate_batch_expiry_fefo(rows_in, forecast, g["as_of_date"].iloc[0],
                                          HORIZON)

        dates = [o["expiry_date"] for o in outs]
        ordered += int(dates == sorted(dates))

        daily_rate = max(0.01, forecast / 28.0)
        last_days = max(o["days_to_expiry"] for o in outs)
        expected_total = max(0.0, g["quantity_on_hand"].sum() - daily_rate * last_days)
        identity += int(abs(sum(o["expected_waste_units"] for o in outs)
                            - expected_total) <= 0.6)

        flagged = any(o["expiry_flag"] in WASTE_FLAGS for o in outs)
        truth = any((r.quantity_on_hand - r.actual_units_window) > 0
                    for r in g.itertuples(index=False))
        agreed += int(flagged == truth)
        groups += 1

    # ---- forecast accuracy on the real history (same 96 predictions) ----
    fc_rows = []
    for (as_of, cat), forecast in forecasts.items():
        actual = float(wide.loc[as_of + ONE_DAY: as_of + HORIZON_TD,
                               CATEGORY_MAPPING[cat]].sum())
        fc_rows.append({"as_of": as_of, "category": cat,
                        "actual": actual, "pred": forecast})
    fc_rows = pd.DataFrame(fc_rows)
    if not fc_rows.empty:
        fc_metrics = metrics(fc_rows["actual"], fc_rows["pred"])

    # ---- breakdowns ----
    parts = []
    for dim, col in [("expiry_window_days", "expiry_window_days"),
                     ("qty_multiplier", "qty_multiplier"),
                     ("category", "category")]:
        for value, g in scored.groupby(col):
            c = confusion(g)
            parts.append({
                "dimension": dim, "value": value, "n": len(g),
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
                "precision": round(c["precision"], 3),
                "recall": round(c["recall"], 3),
                "f1": round(c["f1"], 3),
                "base_rate": round(c["base_rate"], 3),
            })
    breakdown = pd.DataFrame(parts)

    confusion_tbl = pd.DataFrame([
        {"dimension": "overall", "value": "all scenarios", "n": len(scored),
         "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
         "precision": round(m["precision"], 3), "recall": round(m["recall"], 3),
         "f1": round(m["f1"], 3), "accuracy": round(m["accuracy"], 3),
         "base_rate": round(m["base_rate"], 3)},
    ])

    # ---- report ----
    print("=" * 78)
    print("TIER 2  expiry flags on the real history + generated batches")
    print("=" * 78)
    print(f"  forecasts         : {len(forecasts)} "
          f"({batches['as_of'].nunique()} as-of dates x "
          f"{batches['atc_code'].nunique()} categories)")
    print(f"  single-batch cases: {len(scored):,}")
    print(f"  FEFO groups       : {groups} ({len(fefo)} batches)")
    print()
    print(f"  {'':22s} {'truth: waste':>14s} {'truth: cleared':>16s}")
    print(f"  {'engine: at risk':22s} {m['tp']:>14,} {m['fp']:>16,}")
    print(f"  {'engine: safe':22s} {m['fn']:>14,} {m['tn']:>16,}")
    print()
    print(f"  precision {m['precision']:.3f}   recall {m['recall']:.3f}   "
          f"F1 {m['f1']:.3f}   accuracy {m['accuracy']:.3f}")
    print(f"  base rate {m['base_rate']:.3f} — always saying 'safe' would score "
          f"{1 - m['base_rate']:.3f} accuracy")
    if not fc_rows.empty:
        print(f"  real-data forecast WAPE {fc_metrics['wape']:.2f}% "
              f"over {len(fc_rows)} predictions")
    print()
    print("  by expiry window")
    print(breakdown[breakdown["dimension"] == "expiry_window_days"].to_string(index=False))
    print()
    print("  by quantity level (x trailing demand x window)")
    print(breakdown[breakdown["dimension"] == "qty_multiplier"].to_string(index=False))
    print()
    print("  by category")
    print(breakdown[breakdown["dimension"] == "category"].to_string(index=False))
    print()
    print(f"  FEFO: {ordered}/{groups} groups returned in expiry order, "
          f"{identity}/{groups} conserve total demand, "
          f"{agreed}/{groups} group flags match whether any batch spoiled.")

    # ---- artifacts + MLflow ----
    REPORTS.mkdir(exist_ok=True)
    conf_path = REPORTS / "tier2_confusion.csv"
    break_path = REPORTS / "tier2_breakdown.csv"
    scenarios_path = REPORTS / "tier2_scenarios.csv"
    confusion_tbl.to_csv(conf_path, index=False)
    breakdown.to_csv(break_path, index=False)
    scored.drop(columns=["predicted", "actual"]).to_csv(scenarios_path, index=False)

    with mlflow.start_run(run_name="tier2-logic"):
        mlflow.log_params({
            **common_params(LOGIC_DB.name),
            "as_of_dates": int(batches["as_of"].nunique()),
            "single_batch_scenarios": len(scored),
            "fefo_groups": groups,
            "experiment_horizon_days": HORIZON,
            "truth": "quantity on hand - units actually sold in the window",
        })
        mlflow.log_metrics(clean({
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "accuracy": m["accuracy"],
            "base_rate": m["base_rate"],
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            "forecast_wape_pct": fc_metrics["wape"] if not fc_rows.empty else float("nan"),
            "fefo_ordered_share": ordered / groups if groups else float("nan"),
            "fefo_identity_share": identity / groups if groups else float("nan"),
            "fefo_group_agreement": agreed / groups if groups else float("nan"),
        }))
        mlflow.log_artifact(str(conf_path))
        mlflow.log_artifact(str(break_path))
        mlflow.log_artifact(str(scenarios_path))
        print(f"\n  MLflow run logged. Reports: {conf_path.relative_to(ROOT)}, "
              f"{break_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
def main() -> None:
    which = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    mlflow.set_tracking_uri(str(TRACKING))
    mlflow.set_experiment(EXPERIMENT)
    if which == "model":
        evaluate_model()
    elif which == "logic":
        evaluate_logic()
    else:
        raise SystemExit("usage: python src/evaluate.py model|logic")


if __name__ == "__main__":
    main()
