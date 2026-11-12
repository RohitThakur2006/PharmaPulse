"""
seed_db.py — Initializes the database and seeds realistic sample inventory batches.
Clearly marked as synthetic sample inventory for demo and testing.
"""

from pathlib import Path
import pandas as pd

from db import init_db, seed_products, add_batch, get_connection
from features import CATEGORY_MAPPING

SALES_CSV = Path(__file__).resolve().parent.parent / "data" / "salesdaily.csv"

def seed_sales_history(db_path: Path = None):
    """Loads the real Pharma Sales daily history (8 ATC categories) into daily_sales."""
    if db_path is None:
        from db import DB_PATH
        db_path = DB_PATH

    df = pd.read_csv(SALES_CSV, parse_dates=['datum'])
    df = df.melt(
        id_vars='datum',
        value_vars=list(CATEGORY_MAPPING),
        var_name='atc_code',
        value_name='units_sold'
    )

    conn = get_connection(db_path)
    prods = {
        row['atc_code']: row['product_id']
        for row in conn.execute("SELECT product_id, atc_code FROM products;")
    }

    conn.executemany("""
    INSERT OR IGNORE INTO daily_sales (product_id, sale_date, units_sold, unmet_demand)
    VALUES (?, ?, ?, 0);
    """, [
        (prods[atc], date.strftime('%Y-%m-%d'), float(units))
        for atc, date, units in zip(df['atc_code'], df['datum'], df['units_sold'])
    ])

    conn.commit()
    conn.close()
    print(f"Seeded {len(df):,} daily sales rows from {SALES_CSV.name}.")

def seed_sample_inventory(db_path: Path = None):
    """Populates realistic initial inventory batches covering all operational scenarios."""
    if db_path is None:
        from db import DB_PATH
        db_path = DB_PATH

    print("Initializing database schema...")
    init_db(db_path)
    
    print("Seeding products table (8 ATC categories)...")
    seed_products(db_path)

    print("Seeding daily sales history...")
    seed_sales_history(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Get product IDs
    cursor.execute("SELECT product_id, atc_code, name FROM products;")
    products = {row['atc_code']: row['product_id'] for row in cursor.fetchall()}

    print("Seeding sample inventory batches...")

    # Realistic active batches as of end of 2019 / current operations:
    # 1. Paracetamol (N02BE): High volume. One near-expiry batch + one fresh batch
    add_batch(
        batch_id="PARA-2019-B1",
        product_id=products['N02BE'],
        initial_qty=400,
        received_date="2019-06-01",
        expiry_date="2019-10-15", # Expiring in ~15 days
        db_path=db_path
    )
    add_batch(
        batch_id="PARA-2019-B2",
        product_id=products['N02BE'],
        initial_qty=600,
        received_date="2019-08-10",
        expiry_date="2020-08-10", # Safe long expiry
        db_path=db_path
    )

    # 2. Anti-Inflammatory Acetic (M01AB): Healthy optimal stock
    add_batch(
        batch_id="M01AB-2019-01",
        product_id=products['M01AB'],
        initial_qty=160,
        received_date="2019-07-01",
        expiry_date="2020-07-01",
        db_path=db_path
    )

    # 3. Respiratory / Airway (R03): Volatile demand, moderate stock
    add_batch(
        batch_id="R03-2019-B08",
        product_id=products['R03'],
        initial_qty=120,
        received_date="2019-08-15",
        expiry_date="2019-11-01",
        db_path=db_path
    )

    # 4. Antihistamines (R06): Low stock (triggers reorder warning)
    add_batch(
        batch_id="R06-2019-B02",
        product_id=products['R06'],
        initial_qty=30,
        received_date="2019-05-10",
        expiry_date="2020-05-10",
        db_path=db_path
    )

    # 5. Psycholeptics Anxiolytics (N05B): Critical near expiry (surplus)
    add_batch(
        batch_id="N05B-2019-EXPIRE",
        product_id=products['N05B'],
        initial_qty=350,
        received_date="2018-10-01",
        expiry_date="2019-10-08", # Expiring in ~7 days!
        db_path=db_path
    )

    # 6. Analgesics Salicylic (N02BA): normal healthy stock
    add_batch(
        batch_id="N02BA-2019-01",
        product_id=products['N02BA'],
        initial_qty=110,
        received_date="2019-06-15",
        expiry_date="2020-06-15",
        db_path=db_path
    )

    # 7. Psycholeptics Hypnotics (N05C): low-volume category, small stock
    add_batch(
        batch_id="N05C-2019-01",
        product_id=products['N05C'],
        initial_qty=25,
        received_date="2019-07-20",
        expiry_date="2020-07-20",
        db_path=db_path
    )

    conn.close()
    print("Database successfully initialized and seeded with sample inventory!")

if __name__ == "__main__":
    seed_sample_inventory()
