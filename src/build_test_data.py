"""
build_test_data.py -- Builds the two golden-test databases.

Tier 1  pharmacy_test_model.db   Synthetic sales history that resembles the real
                                 one. Measures forecast accuracy on unseen data.

Tier 2  pharmacy_test_logic.db   The REAL sales history plus a generated batch
                                 grid. Measures the expiry flags end to end.

pharmacy.db, src/ and the model artifact are never modified.

    python src/build_test_data.py
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from db import get_connection, init_db, seed_products
from features import CATEGORY_MAPPING

ROOT = Path(__file__).resolve().parent.parent
SALES_CSV = ROOT / "data" / "salesdaily.csv"
REAL_DB = ROOT / "pharmacy.db"
MODEL_DB = ROOT / "pharmacy_test_model.db"
LOGIC_DB = ROOT / "pharmacy_test_logic.db"

GEN_START = "2015-01-01"
GEN_END = "2019-12-31"
SEED = 42

# Tier-2 evaluation grid
AS_OF_DATES = pd.date_range("2018-10-01", "2019-09-01", freq="MS")
EXPIRY_WINDOWS = [3, 7, 10, 14, 21, 28]
MULTIPLIERS = [0.5, 0.9, 1.0, 1.1, 1.5, 3.0]
FEFO_WINDOWS = [14, 21, 28]
FEFO_MULTIPLIERS = [1.5, 2.0, 1.0]

# Acceptance tolerances for the synthetic history
TOL = {"mean": 0.05, "cv": 0.10, "zero": 0.03, "amp": 5.0, "ac": 0.10, "corr": 0.90}


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------
def load_real_history() -> pd.DataFrame:
    """Category-level daily units from the CC0 CSV, indexed by date."""
    df = pd.read_csv(SALES_CSV, parse_dates=["datum"])
    wide = df.set_index("datum")[list(CATEGORY_MAPPING)].sort_index()
    return wide.astype(float)


def closure_days(wide: pd.DataFrame) -> pd.DatetimeIndex:
    """A day is a closure when every category is exactly zero."""
    return wide.index[(wide == 0).all(axis=1)]


# ---------------------------------------------------------------------------
# Generator fitting
# ---------------------------------------------------------------------------
def fit_category(full: pd.Series, open_series: pd.Series) -> dict:
    """
    Fits a hurdle Negative-Binomial profile with weekday / month factors.

    Real demand persists at the WEEKLY level: N02BE correlates 0.44 one day ahead
    and 0.47 seven days ahead, so a daily AR(1) cannot reproduce it. The latent
    level therefore varies per week, not per day.
    """
    mean = float(open_series.mean())
    cv = float(open_series.std()) / mean
    zero_rate = float((open_series == 0).mean())
    ac1 = float(full.autocorr(1))
    ac7 = float(full.autocorr(7))

    dow_mean = open_series.groupby(open_series.index.dayofweek).mean()
    mon_mean = open_series.groupby(open_series.index.month).mean()

    return {
        "mean": mean,
        "cv": cv,
        "zero": zero_rate,
        "ac1": ac1,
        "ac7": ac7,
        "p_sale": max(1e-3, 1.0 - zero_rate),
        "dow_f": (dow_mean / mean).to_numpy(),
        "mon_f": (mon_mean / mean).to_numpy(),
        # How fast the weekly level decays; ac7/ac1 is that decay ratio.
        "phi_w": float(np.clip(ac7 / ac1, 0.30, 0.99)) if ac1 > 0.02 else 0.90,
    }


def _draw_positive(rng: np.random.Generator, mu: np.ndarray, disp: float) -> np.ndarray:
    """Negative Binomial draws conditioned on being at least 1 (hurdle step two)."""
    out = np.zeros_like(mu)
    pending = np.ones(mu.shape, dtype=bool)
    while pending.any():
        idx = np.flatnonzero(pending)
        draws = rng.negative_binomial(disp, disp / (disp + mu[idx]))
        ok = draws > 0
        out[idx[ok]] = draws[ok]
        pending[idx[ok]] = False
    return out


def simulate(par: dict, dates: pd.DatetimeIndex, k: float, s: float,
             disp: float, rng: np.random.Generator) -> np.ndarray:
    """
    One synthetic series: hurdle counts scaled by weekday, month and a weekly
    AR(1) level. `k` sets the mean, `s` the persistence, `disp` the spread.
    """
    n = len(dates)
    week = dates.to_period("W-MON").factorize()[0]
    n_weeks = int(week.max()) + 1 if n else 0

    z = np.empty(n_weeks)
    noise = rng.normal(size=n_weeks)
    z[0] = noise[0]
    decay = np.sqrt(max(0.0, 1 - par["phi_w"] ** 2))
    for w in range(1, n_weeks):
        z[w] = par["phi_w"] * z[w - 1] + decay * noise[w]
    # Centre the latent on this window so its multiplier averages to 1 and the
    # overall level does not depend on which seed drew the path.
    z = z - z.mean()

    base = (par["mean"] * k
            * par["dow_f"][dates.dayofweek.to_numpy()]
            * par["mon_f"][dates.month.to_numpy() - 1])
    mu = base * np.exp(s * z[week] - 0.5 * s * s)

    out = np.zeros(n)
    sale = rng.random(n) < par["p_sale"]
    if sale.any():
        out[sale] = _draw_positive(rng, mu[sale], disp)
    return out


def calibrate(par: dict, dates: pd.DatetimeIndex) -> tuple:
    """
    Solves for (k, s, disp) so the simulated series reproduces the measured
    mean, autocorrelation and CV.

    CV and autocorrelation both depend on `s`, so solving them one after the
    other oscillates. Instead, for every candidate `s` the dispersion is solved
    to match CV, and `s` itself is bisected on the resulting autocorrelation.
    """
    def probe(k, s, disp):
        return simulate(par, dates, k, s, disp, np.random.default_rng(12345))

    def disp_for_cv(s):
        """Spread that makes the CV match, for a given persistence."""
        lo, hi = np.log(0.02), np.log(1e5)
        for _ in range(16):
            mid = float(np.exp((lo + hi) / 2))
            v = probe(1.0, s, mid)
            lo, hi = (np.log(mid), hi) if v.std() / v.mean() > par["cv"] else (lo, np.log(mid))
        return float(np.exp((lo + hi) / 2))

    s = 0.0
    if par["ac1"] >= 0.03:
        lo, hi = 0.0, 0.90
        for _ in range(14):
            mid = (lo + hi) / 2
            v = probe(1.0, mid, disp_for_cv(mid))
            lo, hi = (mid, hi) if np.corrcoef(v[:-1], v[1:])[0, 1] < par["ac1"] else (lo, mid)
        s = (lo + hi) / 2

    disp = disp_for_cv(s)

    lo, hi = 0.05, 8.0
    for _ in range(24):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if probe(mid, s, disp).mean() < par["mean"] else (lo, mid)
    return (lo + hi) / 2, s, disp


# ---------------------------------------------------------------------------
# Closures
# ---------------------------------------------------------------------------
def orthodox_easter(year: int) -> date:
    """Gregorian date of Orthodox Easter (Julian computus + 13-day offset)."""
    a, b, c = year % 4, year % 7, year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    total = d + e + 114
    return date(year, total // 31, total % 31 + 1) + timedelta(days=13)


def closure_dates(start: str, end: str) -> set:
    """The holiday pattern observed in the real data: 5 fixed days per year."""
    days = set()
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        for md in [(1, 1), (1, 7), (5, 1), (12, 19)]:
            days.add(date(year, *md))
        days.add(orthodox_easter(year))
    return {d for d in days if pd.Timestamp(start) <= pd.Timestamp(d) <= pd.Timestamp(end)}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def profile(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-category statistics used to compare real and synthetic histories."""
    closed = (wide == 0).all(axis=1)
    open_days = wide[~closed]
    rows = {}
    for cat in wide.columns:
        s_open = open_days[cat]
        dow = s_open.groupby(s_open.index.dayofweek).mean()
        rows[cat] = {
            "mean": s_open.mean(),
            "cv": s_open.std() / s_open.mean(),
            "zero": (s_open == 0).mean(),
            "amp": 100 * (dow.max() - dow.min()) / dow.mean(),
            "ac1": wide[cat].autocorr(1),
            "ac7": wide[cat].autocorr(7),
        }
    return pd.DataFrame(rows).T


def _amp_tolerance(cv: float, n_days: int) -> float:
    """
    A weekday mean from n/7 draws has sampling error, so its amplitude is noisy
    for low-volume categories. Scale the tolerance to that noise instead of
    applying one fixed number to a 30-unit and a 0.6-unit category alike.
    """
    return max(TOL["amp"], 4.0 * cv / np.sqrt(n_days / 7) * 100)


def _tolerances(r: pd.Series, n_days: int) -> dict:
    """Allowances that account for one 5-year draw being one sample, not a mean."""
    sem = r["cv"] / np.sqrt(n_days)
    return {
        "mean": max(TOL["mean"], 3 * sem),
        "cv": max(TOL["cv"], 5 * sem),
        "zero": max(TOL["zero"], 4 * np.sqrt(max(r["zero"] * (1 - r["zero"]), 0) / n_days)),
        "amp": _amp_tolerance(r["cv"], n_days),
    }


def verify(synth: pd.DataFrame, real: pd.DataFrame) -> None:
    """Prints real vs synthetic and raises if anything drifts out of tolerance."""
    ps, pr = profile(synth), profile(real)
    print(f"\n{'cat':7s} {'mean':>17s} {'CV':>15s} {'zero%':>15s} {'amp%':>15s} "
          f"{'ac1':>13s} {'ac7':>13s}")
    print(f"{'':7s} {'real  synth':>17s} {'real  synth':>15s} {'real  synth':>15s} "
          f"{'real  synth':>15s} {'real  synth':>13s} {'real  synth':>13s}")

    failures = []
    n_days = len(synth)
    for cat in ps.index:
        r, s = pr.loc[cat], ps.loc[cat]
        tol = _tolerances(r, n_days)
        print(f"{cat:7s} {r['mean']:6.1f} {s['mean']:6.1f}   "
              f"{r['cv']:5.2f} {s['cv']:5.2f}   "
              f"{100*r['zero']:5.1f} {100*s['zero']:5.1f}   "
              f"{r['amp']:5.1f} {s['amp']:5.1f}   "
              f"{r['ac1']:5.2f} {s['ac1']:5.2f}   "
              f"{r['ac7']:5.2f} {s['ac7']:5.2f}")

        if abs(s["mean"] / r["mean"] - 1) > tol["mean"]:
            failures.append(f"{cat}: mean {s['mean']:.2f} vs {r['mean']:.2f} "
                            f"(tol {100*tol['mean']:.1f}%)")
        if abs(s["cv"] / r["cv"] - 1) > tol["cv"]:
            failures.append(f"{cat}: CV {s['cv']:.2f} vs {r['cv']:.2f} "
                            f"(tol {100*tol['cv']:.0f}%)")
        if abs(s["zero"] - r["zero"]) > tol["zero"]:
            failures.append(f"{cat}: zero rate {s['zero']:.3f} vs {r['zero']:.3f} "
                            f"(tol {100*tol['zero']:.1f}pt)")

        # A category that is mostly zero (N05C: 67.5%) has no measurable weekday
        # or persistence signal - its correlations are noise. Require only that
        # nothing spurious appears.
        if r["zero"] > 0.40:
            if abs(s["ac1"]) > 0.10 or abs(s["ac7"]) > 0.10:
                failures.append(f"{cat}: spurious autocorrelation "
                                f"({s['ac1']:.2f}, {s['ac7']:.2f})")
        else:
            if abs(s["amp"] - r["amp"]) > tol["amp"]:
                failures.append(f"{cat}: weekday amplitude {s['amp']:.1f} vs "
                                f"{r['amp']:.1f} (tol {tol['amp']:.1f})")
            for ac in ("ac1", "ac7"):
                if abs(s[ac] - r[ac]) > TOL["ac"]:
                    failures.append(f"{cat}: {ac} {s[ac]:.2f} vs {r[ac]:.2f}")

    corr = np.corrcoef(profile_month(real, "N02BE"), profile_month(synth, "N02BE"))[0, 1]
    print(f"\nN02BE month-of-year profile correlation: {corr:.3f} (need > {TOL['corr']})")
    if corr < TOL["corr"]:
        failures.append(f"N02BE month profile correlation {corr:.3f}")

    if (synth.to_numpy() < 0).any():
        failures.append("negative units produced")
    if not np.allclose(synth.to_numpy(), synth.to_numpy().round()):
        failures.append("non-integer units produced")

    n_closed = int((synth == 0).all(axis=1).sum())
    print(f"closure days: real {int((real == 0).all(axis=1).sum())}, synthetic {n_closed} "
          f"(need 20-30)")
    if not 20 <= n_closed <= 30:
        failures.append(f"closure days {n_closed}")

    if failures:
        raise SystemExit("Synthetic history failed verification:\n  " + "\n  ".join(failures))
    print("Verification passed.")


def profile_month(wide: pd.DataFrame, cat: str) -> np.ndarray:
    s = wide[cat]
    return s.groupby(s.index.month).mean().to_numpy()


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------
def build_model_db(real: pd.DataFrame) -> None:
    print("=" * 78)
    print("TIER 1  pharmacy_test_model.db  (synthetic history)")
    print("=" * 78)

    dates = pd.date_range(GEN_START, GEN_END, freq="D")
    closed_real = (real == 0).all(axis=1)
    synth = pd.DataFrame(index=dates, columns=real.columns, dtype=float)

    for i, cat in enumerate(real.columns):
        par = fit_category(real[cat], real.loc[~closed_real, cat])
        k, s, disp = calibrate(par, dates)
        print(f"  {cat:6s} k={k:.3f} s={s:.3f} disp={disp:.1f} phi_w={par['phi_w']:.2f}")
        rng = np.random.default_rng(SEED + i)
        synth[cat] = simulate(par, dates, k, s, disp, rng)

    closed = pd.DatetimeIndex(sorted(
        pd.Timestamp(d) for d in closure_dates(GEN_START, GEN_END) if pd.Timestamp(d) in dates))
    synth.loc[closed, :] = 0
    verify(synth, real)

    if MODEL_DB.exists():
        MODEL_DB.unlink()
    init_db(MODEL_DB, drop_existing=True)
    seed_products(MODEL_DB)

    conn = get_connection(MODEL_DB)
    prods = pd.DataFrame([dict(r) for r in conn.execute(
        "SELECT product_id, atc_code, brand_share FROM products;").fetchall()])
    conn.close()

    rows = []
    for cat in synth.columns:
        sub = prods[prods["atc_code"] == cat]
        for _, p in sub.iterrows():
            rows.append(pd.DataFrame({
                "product_id": int(p["product_id"]),
                "sale_date": dates.strftime("%Y-%m-%d"),
                "units_sold": (synth[cat].to_numpy() * float(p["brand_share"])).round(2),
            }))

    all_rows = pd.concat(rows, ignore_index=True)
    conn = get_connection(MODEL_DB)
    conn.executemany(
        "INSERT INTO daily_sales (product_id, sale_date, units_sold, unmet_demand) "
        "VALUES (?, ?, ?, 0);",
        list(all_rows.itertuples(index=False, name=None)),
    )
    conn.commit()
    conn.close()

    print(f"\n  days          : {len(dates):,}  ({GEN_START} -> {GEN_END})")
    print(f"  categories    : {len(synth.columns)}")
    print(f"  daily_sales   : {len(all_rows):,} rows "
          f"({len(prods)} products x {len(dates):,} days)")
    print(f"  written       : {MODEL_DB.name}")


# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------
def copy_real_sales() -> int:
    src = get_connection(REAL_DB)
    rows = [tuple(r) for r in src.execute(
        "SELECT product_id, sale_date, units_sold, unmet_demand FROM daily_sales;").fetchall()]
    src.close()

    dst = get_connection(LOGIC_DB)
    dst.executemany(
        "INSERT OR REPLACE INTO daily_sales (product_id, sale_date, units_sold, unmet_demand) "
        "VALUES (?, ?, ?, ?);", rows)
    dst.commit()
    dst.close()
    return len(rows)


def build_logic_db(real: pd.DataFrame) -> None:
    print()
    print("=" * 78)
    print("TIER 2  pharmacy_test_logic.db  (real history + generated batches)")
    print("=" * 78)

    if LOGIC_DB.exists():
        LOGIC_DB.unlink()
    init_db(LOGIC_DB, drop_existing=True)
    seed_products(LOGIC_DB)
    n_sales = copy_real_sales()

    conn = get_connection(LOGIC_DB)
    conn.execute("""
    CREATE TABLE test_batches (
        batch_id TEXT PRIMARY KEY,
        scenario_id TEXT NOT NULL,
        as_of_date DATE NOT NULL,
        atc_code TEXT NOT NULL,
        product_id INTEGER NOT NULL,
        quantity_on_hand REAL NOT NULL,
        received_date DATE NOT NULL,
        expiry_date DATE NOT NULL,
        qty_multiplier REAL NOT NULL,
        expiry_window_days INTEGER NOT NULL,
        actual_units_window REAL NOT NULL
    );
    """)

    prods = pd.DataFrame([dict(r) for r in conn.execute(
        "SELECT product_id, atc_code, brand_share FROM products;").fetchall()])

    batches = []
    for cat in real.columns:
        lead = prods[prods["atc_code"] == cat].sort_values("brand_share", ascending=False).iloc[0]
        series = real[cat]

        for as_of in AS_OF_DATES:
            trailing = series.loc[as_of - timedelta(days=364):as_of]
            rate = float(trailing.mean())
            if rate <= 0:
                continue
            for window in EXPIRY_WINDOWS:
                for mult in MULTIPLIERS:
                    batches.append(_batch_row("G", cat, lead, as_of, window, mult, rate, series))
            for window, mult in zip(FEFO_WINDOWS, FEFO_MULTIPLIERS):
                batches.append(_batch_row("F", cat, lead, as_of, window, mult, rate, series))

    conn.executemany("""
    INSERT INTO test_batches (batch_id, scenario_id, as_of_date, atc_code, product_id,
                              quantity_on_hand, received_date, expiry_date, qty_multiplier,
                              expiry_window_days, actual_units_window)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, batches)
    conn.commit()
    conn.close()

    qty = pd.Series([b[5] for b in batches])
    print(f"  daily_sales   : {n_sales:,} rows (copied from pharmacy.db, unchanged)")
    print(f"  test_batches  : {len(batches):,} rows")
    print(f"  as_of dates   : {len(AS_OF_DATES)} "
          f"({AS_OF_DATES[0].date()} -> {AS_OF_DATES[-1].date()})")
    print(f"  expiry windows: {EXPIRY_WINDOWS} core, {FEFO_WINDOWS} staggered-FEFO")
    print(f"  quantity x    : {MULTIPLIERS}")
    print(f"  quantity range: {qty.min():.0f} .. {qty.max():.0f} units")
    print(f"  written       : {LOGIC_DB.name}")


def _batch_row(tag, cat, lead, as_of, window, mult, rate, series) -> tuple:
    """
    Sizes the batch from trailing demand, then records what actually sold.

    `scenario_id` says how the batch is evaluated: a 'G' row is a scenario of
    its own (one batch, one demand pool), while the three 'F' rows for a given
    category and date share one id and must be passed to the engine together so
    FEFO ordering is exercised.
    """
    qty = max(1.0, round(mult * rate * window))
    expiry = as_of + timedelta(days=window)
    actual = float(series.loc[as_of + timedelta(days=1):as_of + timedelta(days=window)].sum())
    batch_id = f"T2-{tag}-{cat}-{as_of:%Y%m%d}-{window:02d}-{int(mult*100):03d}"
    scenario = batch_id if tag == "G" else f"T2-F-{cat}-{as_of:%Y%m%d}"
    return (
        batch_id,
        scenario,
        as_of.strftime("%Y-%m-%d"),
        cat,
        int(lead["product_id"]),
        qty,
        (as_of - timedelta(days=30)).strftime("%Y-%m-%d"),
        expiry.strftime("%Y-%m-%d"),
        mult,
        window,
        actual,
    )


# ---------------------------------------------------------------------------
def main() -> None:
    real = load_real_history()
    build_model_db(real)
    build_logic_db(real)
    print("\nDone. pharmacy.db was not modified.")


if __name__ == "__main__":
    main()
