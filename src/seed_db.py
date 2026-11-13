"""
seed_db.py — Initializes the database and seeds realistic sample inventory batches
for 38 essential Indian household medicines mapped to the 8 ATC categories.
"""

from pathlib import Path
import pandas as pd

from db import init_db, seed_products, add_batch, get_connection
from features import CATEGORY_MAPPING

SALES_CSV = Path(__file__).resolve().parent.parent / "data" / "salesdaily.csv"

def seed_sales_history(db_path: Path = None):
    """
    Loads daily history from salesdaily.csv. Apportions category sales
    among constituent Indian brand medicines by their brand_share.
    Sum of brand sales equals the exact category sales.
    """
    if db_path is None:
        from db import DB_PATH
        db_path = DB_PATH

    df = pd.read_csv(SALES_CSV, parse_dates=['datum'])
    df_melt = df.melt(
        id_vars='datum',
        value_vars=list(CATEGORY_MAPPING),
        var_name='atc_code',
        value_name='units_sold'
    )

    conn = get_connection(db_path)
    prods = conn.execute("SELECT product_id, atc_code, brand_share FROM products;").fetchall()
    conn.close()

    prods_df = pd.DataFrame([dict(r) for r in prods])

    merged = pd.merge(df_melt, prods_df, on='atc_code')
    merged['brand_units_sold'] = merged['units_sold'] * merged['brand_share']
    merged['sale_date'] = merged['datum'].dt.strftime('%Y-%m-%d')

    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.executemany("""
    INSERT OR REPLACE INTO daily_sales (product_id, sale_date, units_sold, unmet_demand)
    VALUES (?, ?, ?, 0);
    """, [
        (int(pid), s_date, round(float(u), 2))
        for pid, s_date, u in zip(merged['product_id'], merged['sale_date'], merged['brand_units_sold'])
    ])

    conn.commit()
    conn.close()
    print(f"Seeded {len(merged):,} brand-level daily sales records from {SALES_CSV.name}.")

def seed_sample_inventory(db_path: Path = None):
    """Populates realistic initial inventory batches covering Indian household medicines."""
    if db_path is None:
        from db import DB_PATH
        db_path = DB_PATH

    print("Re-initializing database schema (drop_existing=True)...")
    init_db(db_path, drop_existing=True)
    
    print("Seeding products table (38 Indian household medicines across 8 ATC categories)...")
    seed_products(db_path)

    print("Seeding daily sales history with brand share apportionment...")
    seed_sales_history(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT product_id, name FROM products;")
    p_map = {row['name']: row['product_id'] for row in cursor.fetchall()}
    conn.close()

    print("Seeding diverse sample inventory batches for testing...")

    # --- CATEGORY 1: N02BE (Paracetamol & Anilides) ---
    # Dolo 650: Batch 1 near-expiry (waste risk), Batch 2 fresh
    add_batch("LOT-DOLO-19A", p_map['Dolo 650mg Tablet'], 320, "2019-06-01", "2019-10-18", db_path)
    add_batch("LOT-DOLO-19B", p_map['Dolo 650mg Tablet'], 550, "2019-08-10", "2020-09-30", db_path)
    # Crocin Advance: Optimal
    add_batch("LOT-CROC-19A", p_map['Crocin Advance 500mg'], 240, "2019-07-15", "2020-08-15", db_path)
    # Calpol 500: Healthy
    add_batch("LOT-CALP-19A", p_map['Calpol 500mg Tablet'], 160, "2019-07-20", "2020-07-20", db_path)
    # Pacimol 650: Already expired batch (discard test)
    add_batch("LOT-PACI-EXP", p_map['Pacimol 650mg Tablet'], 85, "2018-09-01", "2019-09-15", db_path)
    add_batch("LOT-PACI-19A", p_map['Pacimol 650mg Tablet'], 110, "2019-08-01", "2020-10-10", db_path)
    # Sumo L: Low stock
    add_batch("LOT-SUMO-19A", p_map['Sumo L 250mg Suspension'], 30, "2019-08-15", "2020-06-30", db_path)

    # --- CATEGORY 2: M01AE (Anti-Inflammatory Propionic) ---
    # Combiflam: Critical low stock (reorder warning)
    add_batch("LOT-COMB-19A", p_map['Combiflam Tablet'], 40, "2019-08-20", "2020-06-30", db_path)
    # Brufen 400: Optimal
    add_batch("LOT-BRUF-19A", p_map['Brufen 400mg Tablet'], 120, "2019-07-01", "2020-11-20", db_path)
    # Ibugesic Plus: Healthy
    add_batch("LOT-IBUG-19A", p_map['Ibugesic Plus Suspension'], 50, "2019-08-01", "2020-08-01", db_path)
    # Flexon: Near expiry
    add_batch("LOT-FLEX-19A", p_map['Flexon Tablet'], 75, "2018-11-01", "2019-10-14", db_path)
    # Ketorol DT: Low stock
    add_batch("LOT-KETO-19A", p_map['Ketorol DT 10mg'], 25, "2019-08-10", "2020-09-15", db_path)

    # --- CATEGORY 3: M01AB (Anti-Inflammatory Acetic) ---
    # Voveran 50: Optimal
    add_batch("LOT-VOV50-19A", p_map['Voveran 50mg Tablet'], 110, "2019-07-01", "2020-07-01", db_path)
    # Voveran SR 100: Healthy
    add_batch("LOT-VOVSR-19A", p_map['Voveran SR 100mg'], 70, "2019-07-10", "2020-07-10", db_path)
    # Voltaren Gel: Optimal
    add_batch("LOT-VOLT-19A", p_map['Voltaren Emulgel (30g)'], 40, "2019-06-25", "2020-09-15", db_path)
    # Diclogesic: Healthy
    add_batch("LOT-DICL-19A", p_map['Diclogesic Tablet'], 65, "2019-07-05", "2020-08-10", db_path)
    # Dynapar AQ: Expired batch test
    add_batch("LOT-DYNA-EXP", p_map['Dynapar AQ Injection 1ml'], 50, "2018-08-01", "2019-08-30", db_path)
    add_batch("LOT-DYNA-19A", p_map['Dynapar AQ Injection 1ml'], 40, "2019-08-15", "2020-12-15", db_path)

    # --- CATEGORY 4: N02BA (Analgesics Salicylic) ---
    # Disprin: Critical low stock
    add_batch("LOT-DISP-19A", p_map['Disprin Regular 350mg'], 20, "2019-07-10", "2020-06-15", db_path)
    # Ecosprin 75: Optimal
    add_batch("LOT-ECO75-19A", p_map['Ecosprin 75mg Tablet'], 95, "2019-07-01", "2020-10-15", db_path)
    # Ecosprin 150: Healthy
    add_batch("LOT-ECO150-19A", p_map['Ecosprin 150mg Tablet'], 55, "2019-07-15", "2020-11-15", db_path)
    # Delisprin: Low stock
    add_batch("LOT-DELI-19A", p_map['Delisprin 75mg Tablet'], 18, "2019-08-01", "2020-07-20", db_path)

    # --- CATEGORY 5: R06 (Antihistamines Systemic) ---
    # Allegra 120: Expiry risk (expiring in 9 days)
    add_batch("LOT-ALLG-19A", p_map['Allegra 120mg Tablet'], 90, "2018-10-15", "2019-10-12", db_path)
    add_batch("LOT-ALLG-19B", p_map['Allegra 120mg Tablet'], 130, "2019-08-01", "2020-11-01", db_path)
    # Cetzine 10: Optimal
    add_batch("LOT-CETZ-19A", p_map['Cetzine 10mg Tablet'], 140, "2019-06-10", "2020-12-15", db_path)
    # Avil 25: Low stock
    add_batch("LOT-AVIL-19A", p_map['Avil 25mg Tablet'], 25, "2019-05-10", "2020-05-10", db_path)
    # Levocet 5: Healthy
    add_batch("LOT-LEVO-19A", p_map['Levocet 5mg Tablet'], 80, "2019-07-01", "2020-09-01", db_path)
    # Montair LC: Optimal
    add_batch("LOT-MONT-19A", p_map['Montair LC Tablet'], 105, "2019-08-01", "2020-10-30", db_path)

    # --- CATEGORY 6: R03 (Respiratory & Airway) ---
    # Asthalin Inhaler: Critical low stock (reorder warning)
    add_batch("LOT-ASTH-19A", p_map['Asthalin 100mcg Inhaler'], 22, "2019-08-01", "2020-08-01", db_path)
    # Deriphyllin Retard: Healthy
    add_batch("LOT-DERI-19A", p_map['Deriphyllin Retard 150mg'], 65, "2019-07-20", "2020-09-20", db_path)
    # Budecort 200: Optimal
    add_batch("LOT-BUDE-19A", p_map['Budecort 200 Inhaler'], 50, "2019-07-15", "2020-10-15", db_path)
    # Foracort 200: Safe
    add_batch("LOT-FORA-19A", p_map['Foracort 200 Rotacaps'], 40, "2019-08-01", "2020-12-01", db_path)
    # Aerocort: Low stock
    add_batch("LOT-AERO-19A", p_map['Aerocort Inhaler'], 15, "2019-08-10", "2020-07-10", db_path)

    # --- CATEGORY 7: N05B (Psycholeptics Anxiolytics) ---
    # Alprax 0.25: Batch 1 is expired on 2019-10-08, Batch 2 is fresh
    add_batch("LOT-ALPX-19A", p_map['Alprax 0.25mg Tablet'], 180, "2018-10-01", "2019-10-08", db_path)
    add_batch("LOT-ALPX-19B", p_map['Alprax 0.25mg Tablet'], 120, "2019-08-15", "2020-08-15", db_path)
    # Alprax 0.5: Optimal
    add_batch("LOT-ALPX5-19A", p_map['Alprax 0.5mg Tablet'], 90, "2019-07-10", "2020-09-10", db_path)
    # Restyl 0.25: Moderate
    add_batch("LOT-REST-19A", p_map['Restyl 0.25mg Tablet'], 45, "2019-08-01", "2020-10-01", db_path)
    # Valium 5mg: Low stock
    add_batch("LOT-VALI-19A", p_map['Valium 5mg Tablet'], 20, "2019-06-15", "2020-08-15", db_path)
    # Librium 10mg: Safe
    add_batch("LOT-LIBR-19A", p_map['Librium 10mg Tablet'], 35, "2019-07-20", "2020-09-20", db_path)

    # --- CATEGORY 8: N05C (Psycholeptics Hypnotics & Sedatives) ---
    # Zolfresh 10: Optimal
    add_batch("LOT-ZOLF-19A", p_map['Zolfresh 10mg Tablet'], 35, "2019-07-20", "2020-07-20", db_path)
    # Nitrosun 10: Healthy
    add_batch("LOT-NITR-19A", p_map['Nitrosun 10mg Tablet'], 20, "2019-06-20", "2020-06-20", db_path)
    # Hypnite 2: Safe
    add_batch("LOT-HYPN-19A", p_map['Hypnite 2mg Tablet'], 25, "2019-07-01", "2020-09-01", db_path)
    # Zopicon 7.5: Low stock
    add_batch("LOT-ZOPI-19A", p_map['Zopicon 7.5mg Tablet'], 12, "2019-08-01", "2020-08-01", db_path)

    print("Database successfully initialized and seeded with 38 Indian household medicines!")

if __name__ == "__main__":
    seed_sample_inventory()
