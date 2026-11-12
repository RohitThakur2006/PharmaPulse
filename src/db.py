"""
db.py — Database schema and data access layer (SQLite).
Provides relational schema: products, inventory_batches, daily_sales, and stock_movements.
Includes atomic transactions and audit logging.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_PATH = Path(__file__).resolve().parent.parent / "pharmacy.db"

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Returns an active SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Path = DB_PATH):
    """Initializes all tables if they do not exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 1. Products Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        atc_code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        category_full_name TEXT NOT NULL
    );
    """)

    # 2. Inventory Batches Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inventory_batches (
        batch_id TEXT PRIMARY KEY,
        product_id INTEGER NOT NULL,
        initial_quantity REAL NOT NULL,
        quantity_on_hand REAL NOT NULL,
        received_date DATE NOT NULL,
        expiry_date DATE NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );
    """)

    # 3. Daily Sales History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_sales (
        sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL,
        sale_date DATE NOT NULL,
        units_sold REAL NOT NULL,
        unmet_demand REAL DEFAULT 0,
        FOREIGN KEY (product_id) REFERENCES products(product_id),
        UNIQUE(product_id, sale_date)
    );
    """)

    # 4. Stock Movements (Audit Ledger)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_movements (
        movement_id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL,
        movement_date DATE NOT NULL,
        movement_type TEXT NOT NULL, -- 'DELIVERY', 'SALE', 'EXPIRED_WASTE', 'ADJUSTMENT'
        quantity_change REAL NOT NULL, -- positive for delivery, negative for sale/waste
        notes TEXT,
        FOREIGN KEY (batch_id) REFERENCES inventory_batches(batch_id)
    );
    """)

    conn.commit()
    conn.close()

def seed_products(db_path: Path = DB_PATH):
    """Seeds the 8 reference ATC medicine categories."""
    products = [
        ('M01AB', 'Anti-Inflammatory (Acetic)', 'M01AB_Anti_Inflammatory_Acetic'),
        ('M01AE', 'Anti-Inflammatory (Propionic)', 'M01AE_Anti_Inflammatory_Propionic'),
        ('N02BA', 'Analgesics (Salicylic)', 'N02BA_Analgesics_Salicylic'),
        ('N02BE', 'Analgesics (Paracetamol & Anilides)', 'N02BE_Analgesics_Paracetamol_Anilides'),
        ('N05B',  'Psycholeptics (Anxiolytics)', 'N05B_Psycholeptics_Anxiolytics'),
        ('N05C',  'Psycholeptics (Hypnotics & Sedatives)', 'N05C_Psycholeptics_Hypnotics_Sedatives'),
        ('R03',   'Respiratory & Airway', 'R03_Respiratory_Airway_Diseases'),
        ('R06',   'Antihistamines (Systemic)', 'R06_Antihistamines_Systemic')
    ]

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT OR IGNORE INTO products (atc_code, name, category_full_name)
    VALUES (?, ?, ?);
    """, products)
    conn.commit()
    conn.close()

def log_sale_atomic(product_id: int, sale_date: str, units_requested: float, db_path: Path = DB_PATH) -> Dict[str, Any]:
    """
    Logs a sale atomically:
    1. Finds active batches sorted by FEFO (earliest expiry first).
    2. Deducts stock across batches.
    3. If units_requested > total stock, records remaining as unmet_demand.
    4. Updates daily_sales and logs stock_movements ledger.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    try:
        # Fetch active batches in FEFO order (earliest expiry date first)
        cursor.execute("""
        SELECT batch_id, quantity_on_hand, expiry_date 
        FROM inventory_batches 
        WHERE product_id = ? AND quantity_on_hand > 0 
        ORDER BY expiry_date ASC;
        """, (product_id,))
        batches = cursor.fetchall()

        units_left_to_fill = units_requested
        actual_units_dispensed = 0.0

        for b in batches:
            if units_left_to_fill <= 0:
                break
            
            b_id = b['batch_id']
            b_stock = b['quantity_on_hand']

            deduct = min(b_stock, units_left_to_fill)
            new_stock = b_stock - deduct
            
            # Update batch
            cursor.execute("UPDATE inventory_batches SET quantity_on_hand = ? WHERE batch_id = ?", (new_stock, b_id))
            
            # Record audit movement
            cursor.execute("""
            INSERT INTO stock_movements (batch_id, movement_date, movement_type, quantity_change, notes)
            VALUES (?, ?, 'SALE', ?, 'Daily POS dispensing');
            """, (b_id, sale_date, -deduct))

            units_left_to_fill -= deduct
            actual_units_dispensed += deduct

        unmet = max(0.0, units_left_to_fill)

        # Upsert into daily_sales
        cursor.execute("""
        INSERT INTO daily_sales (product_id, sale_date, units_sold, unmet_demand)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(product_id, sale_date) DO UPDATE SET
            units_sold = units_sold + excluded.units_sold,
            unmet_demand = unmet_demand + excluded.unmet_demand;
        """, (product_id, sale_date, actual_units_dispensed, unmet))

        conn.commit()
        return {
            'dispensed': actual_units_dispensed,
            'unmet_demand': unmet
        }

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def add_batch(batch_id: str, product_id: int, initial_qty: float, received_date: str, expiry_date: str, db_path: Path = DB_PATH):
    """Inserts a new batch delivery and records audit movement."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO inventory_batches (batch_id, product_id, initial_quantity, quantity_on_hand, received_date, expiry_date)
        VALUES (?, ?, ?, ?, ?, ?);
        """, (batch_id, product_id, initial_qty, initial_qty, received_date, expiry_date))

        cursor.execute("""
        INSERT INTO stock_movements (batch_id, movement_date, movement_type, quantity_change, notes)
        VALUES (?, ?, 'DELIVERY', ?, 'Supplier delivery receipt');
        """, (batch_id, received_date, initial_qty))

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
