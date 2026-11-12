"""
seed_db.py — Initializes the database and seeds realistic sample inventory batches.
Clearly marked as synthetic sample inventory for demo and testing.
"""

from pathlib import Path
from db import init_db, seed_products, add_batch, get_connection

def seed_sample_inventory(db_path: Path = None):
    """Populates realistic initial inventory batches covering all operational scenarios."""
    if db_path is None:
        from db import DB_PATH
        db_path = DB_PATH

    print("Initializing database schema...")
    init_db(db_path)
    
    print("Seeding products table (8 ATC categories)...")
    seed_products(db_path)

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

    conn.close()
    print("✅ Database successfully initialized and seeded with sample inventory!")

if __name__ == "__main__":
    seed_sample_inventory()
