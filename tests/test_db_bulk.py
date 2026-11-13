"""
test_db_bulk.py — Unit test verifying log_sales_bulk_atomic and get_daily_sales_for_date.
"""

import pytest
import sqlite3
import shutil
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from db import init_db, seed_products, add_batch, log_sales_bulk_atomic, get_daily_sales_for_date

TEST_DB_PATH = Path(__file__).resolve().parent.parent / "pharmacy_test_bulk.db"

@pytest.fixture(autouse=True)
def setup_test_db():
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    init_db(TEST_DB_PATH)
    seed_products(TEST_DB_PATH)

    # Add 2 batches for Product 1 (Dolo 650):
    # Batch A expires earlier (2020-05-01, 30 units)
    # Batch B expires later (2020-08-01, 50 units)
    add_batch('LOT-DOLO-EARLY', 1, 30.0, '2019-10-01', '2020-05-01', db_path=TEST_DB_PATH)
    add_batch('LOT-DOLO-LATER', 1, 50.0, '2019-10-01', '2020-08-01', db_path=TEST_DB_PATH)

    # Add 1 batch for Product 6 (Combiflam, 40 units)
    add_batch('LOT-COMB-TEST', 6, 40.0, '2019-10-01', '2020-06-01', db_path=TEST_DB_PATH)

    yield

    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()

def test_bulk_dispense_multi_medicine():
    # Dispense 40 units of Dolo (should fully consume LOT-DOLO-EARLY 30u + take 10u from LOT-DOLO-LATER)
    # Dispense 15 units of Combiflam (should take 15u from LOT-COMB-TEST)
    sales_payload = [
        {'product_id': 1, 'product_name': 'Dolo 650mg Tablet', 'units': 40.0},
        {'product_id': 6, 'product_name': 'Combiflam Tablet', 'units': 15.0}
    ]

    res = log_sales_bulk_atomic(sales_payload, sale_date='2019-10-08', db_path=TEST_DB_PATH)
    assert res['total_medicines'] == 2
    assert res['total_dispensed'] == 55.0
    assert res['total_unmet'] == 0.0

    # Verify Dolo batches in DB
    conn = sqlite3.connect(TEST_DB_PATH)
    cur = conn.cursor()
    b_early = cur.execute("SELECT quantity_on_hand FROM inventory_batches WHERE batch_id='LOT-DOLO-EARLY'").fetchone()[0]
    b_later = cur.execute("SELECT quantity_on_hand FROM inventory_batches WHERE batch_id='LOT-DOLO-LATER'").fetchone()[0]
    b_comb = cur.execute("SELECT quantity_on_hand FROM inventory_batches WHERE batch_id='LOT-COMB-TEST'").fetchone()[0]
    conn.close()

    assert b_early == 0.0  # Completely consumed
    assert b_later == 40.0 # 50 - 10 = 40
    assert b_comb == 25.0  # 40 - 15 = 25

    # Verify get_daily_sales_for_date
    day_sales = get_daily_sales_for_date('2019-10-08', db_path=TEST_DB_PATH)
    assert len(day_sales) == 2
    dolo_sale = next(s for s in day_sales if s['product_id'] == 1)
    comb_sale = next(s for s in day_sales if s['product_id'] == 6)
    assert dolo_sale['units_sold'] == 40.0
    assert comb_sale['units_sold'] == 15.0
