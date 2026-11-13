"""
db.py — Database schema and data access layer (SQLite).
Provides relational schema: products (Indian household medicines),
inventory_batches, daily_sales, and stock_movements.
Includes atomic transactions and audit logging.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_PATH = Path(__file__).resolve().parent.parent / "pharmacy.db"

# 38 Essential Indian Household Medicines across the 8 ATC Categories
INDIAN_HOUSEHOLD_PRODUCTS = [
    # 1. N02BE: Paracetamol & Anilides (Fever, pain relief)
    ('Dolo 650mg Tablet', 'Paracetamol 650mg', 'N02BE', 'N02BE_Analgesics_Paracetamol_Anilides', 0.40),
    ('Crocin Advance 500mg', 'Paracetamol Fast-Release 500mg', 'N02BE', 'N02BE_Analgesics_Paracetamol_Anilides', 0.25),
    ('Calpol 500mg Tablet', 'Paracetamol Pediatric/Adult 500mg', 'N02BE', 'N02BE_Analgesics_Paracetamol_Anilides', 0.15),
    ('Pacimol 650mg Tablet', 'Paracetamol 650mg', 'N02BE', 'N02BE_Analgesics_Paracetamol_Anilides', 0.12),
    ('Sumo L 250mg Suspension', 'Paracetamol Pediatric Oral Suspension', 'N02BE', 'N02BE_Analgesics_Paracetamol_Anilides', 0.08),

    # 2. M01AE: Anti-Inflammatory Propionic (Body ache, dental pain, arthritis)
    ('Combiflam Tablet', 'Ibuprofen 400mg + Paracetamol 325mg', 'M01AE', 'M01AE_Anti_Inflammatory_Propionic', 0.45),
    ('Brufen 400mg Tablet', 'Ibuprofen 400mg', 'M01AE', 'M01AE_Anti_Inflammatory_Propionic', 0.25),
    ('Ibugesic Plus Suspension', 'Ibuprofen 100mg + Paracetamol 162.5mg', 'M01AE', 'M01AE_Anti_Inflammatory_Propionic', 0.12),
    ('Flexon Tablet', 'Ibuprofen 400mg + Paracetamol 500mg', 'M01AE', 'M01AE_Anti_Inflammatory_Propionic', 0.10),
    ('Ketorol DT 10mg', 'Ketorolac Tromethamine 10mg (Dental Pain)', 'M01AE', 'M01AE_Anti_Inflammatory_Propionic', 0.08),

    # 3. M01AB: Anti-Inflammatory Acetic (Severe joint pain, sprains, post-operative)
    ('Voveran 50mg Tablet', 'Diclofenac Sodium 50mg', 'M01AB', 'M01AB_Anti_Inflammatory_Acetic', 0.40),
    ('Voveran SR 100mg', 'Diclofenac Sustained Release 100mg', 'M01AB', 'M01AB_Anti_Inflammatory_Acetic', 0.25),
    ('Voltaren Emulgel (30g)', 'Diclofenac Diethylamine Topical Gel', 'M01AB', 'M01AB_Anti_Inflammatory_Acetic', 0.15),
    ('Diclogesic Tablet', 'Diclofenac 50mg + Paracetamol 325mg', 'M01AB', 'M01AB_Anti_Inflammatory_Acetic', 0.12),
    ('Dynapar AQ Injection 1ml', 'Diclofenac Sodium Injection 75mg/ml', 'M01AB', 'M01AB_Anti_Inflammatory_Acetic', 0.08),

    # 4. N02BA: Analgesics Salicylic (Cardio protection, headache, fever)
    ('Disprin Regular 350mg', 'Soluble Effervescent Aspirin 350mg', 'N02BA', 'N02BA_Analgesics_Salicylic', 0.40),
    ('Ecosprin 75mg Tablet', 'Enteric-Coated Aspirin 75mg (Cardio)', 'N02BA', 'N02BA_Analgesics_Salicylic', 0.35),
    ('Ecosprin 150mg Tablet', 'Enteric-Coated Aspirin 150mg', 'N02BA', 'N02BA_Analgesics_Salicylic', 0.15),
    ('Delisprin 75mg Tablet', 'Aspirin Gastro-Resistant 75mg', 'N02BA', 'N02BA_Analgesics_Salicylic', 0.10),

    # 5. R06: Antihistamines Systemic (Allergies, cold, sneezing, runny nose)
    ('Allegra 120mg Tablet', 'Fexofenadine HCl 120mg', 'R06', 'R06_Antihistamines_Systemic', 0.35),
    ('Cetzine 10mg Tablet', 'Cetirizine Dihydrochloride 10mg', 'R06', 'R06_Antihistamines_Systemic', 0.25),
    ('Avil 25mg Tablet', 'Pheniramine Maleate 25mg', 'R06', 'R06_Antihistamines_Systemic', 0.18),
    ('Levocet 5mg Tablet', 'Levocetirizine Dihydrochloride 5mg', 'R06', 'R06_Antihistamines_Systemic', 0.12),
    ('Montair LC Tablet', 'Montelukast 10mg + Levocetirizine 5mg', 'R06', 'R06_Antihistamines_Systemic', 0.10),

    # 6. R03: Respiratory & Airway (Asthma, wheezing, bronchospasm, inhalers)
    ('Asthalin 100mcg Inhaler', 'Salbutamol Inhaler (200 actuations)', 'R03', 'R03_Respiratory_Airway_Diseases', 0.38),
    ('Deriphyllin Retard 150mg', 'Theophylline + Etophylline', 'R03', 'R03_Respiratory_Airway_Diseases', 0.24),
    ('Budecort 200 Inhaler', 'Budesonide 200mcg Inhalation Aerosol', 'R03', 'R03_Respiratory_Airway_Diseases', 0.18),
    ('Foracort 200 Rotacaps', 'Formoterol 6mcg + Budesonide 200mcg', 'R03', 'R03_Respiratory_Airway_Diseases', 0.12),
    ('Aerocort Inhaler', 'Levosalbutamol + Beclomethasone', 'R03', 'R03_Respiratory_Airway_Diseases', 0.08),

    # 7. N05B: Psycholeptics Anxiolytics (Anxiety, panic attacks, tension)
    ('Alprax 0.25mg Tablet', 'Alprazolam 0.25mg', 'N05B', 'N05B_Psycholeptics_Anxiolytics', 0.40),
    ('Alprax 0.5mg Tablet', 'Alprazolam 0.5mg', 'N05B', 'N05B_Psycholeptics_Anxiolytics', 0.25),
    ('Restyl 0.25mg Tablet', 'Alprazolam 0.25mg', 'N05B', 'N05B_Psycholeptics_Anxiolytics', 0.15),
    ('Valium 5mg Tablet', 'Diazepam 5mg', 'N05B', 'N05B_Psycholeptics_Anxiolytics', 0.12),
    ('Librium 10mg Tablet', 'Chlordiazepoxide 10mg', 'N05B', 'N05B_Psycholeptics_Anxiolytics', 0.08),

    # 8. N05C: Psycholeptics Hypnotics & Sedatives (Insomnia, sleep disturbances)
    ('Zolfresh 10mg Tablet', 'Zolpidem Tartrate 10mg', 'N05C', 'N05C_Psycholeptics_Hypnotics_Sedatives', 0.45),
    ('Nitrosun 10mg Tablet', 'Nitrazepam 10mg', 'N05C', 'N05C_Psycholeptics_Hypnotics_Sedatives', 0.25),
    ('Hypnite 2mg Tablet', 'Eszopiclone 2mg', 'N05C', 'N05C_Psycholeptics_Hypnotics_Sedatives', 0.18),
    ('Zopicon 7.5mg Tablet', 'Zopiclone 7.5mg', 'N05C', 'N05C_Psycholeptics_Hypnotics_Sedatives', 0.12)
]

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Returns an active SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Path = DB_PATH, drop_existing: bool = False):
    """Initializes all tables if they do not exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    if drop_existing:
        cursor.execute("PRAGMA foreign_keys = OFF;")
        cursor.execute("DROP TABLE IF EXISTS stock_movements;")
        cursor.execute("DROP TABLE IF EXISTS daily_sales;")
        cursor.execute("DROP TABLE IF EXISTS inventory_batches;")
        cursor.execute("DROP TABLE IF EXISTS products;")
        cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Products Table (Indian household brands mapped to ATC codes)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        generic_name TEXT NOT NULL,
        atc_code TEXT NOT NULL,
        category_full_name TEXT NOT NULL,
        brand_share REAL NOT NULL DEFAULT 0.10
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
    """Seeds the 38 Indian household medicines grouped across the 8 ATC categories."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT OR REPLACE INTO products (name, generic_name, atc_code, category_full_name, brand_share)
    VALUES (?, ?, ?, ?, ?);
    """, INDIAN_HOUSEHOLD_PRODUCTS)
    conn.commit()
    conn.close()

def add_product(name: str, generic_name: str, atc_code: str, category_full_name: str, brand_share: float = 0.10, db_path: Path = DB_PATH) -> int:
    """Inserts a new medicine brand into the products table and returns its product_id."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO products (name, generic_name, atc_code, category_full_name, brand_share)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(name) DO UPDATE SET
        generic_name = excluded.generic_name,
        atc_code = excluded.atc_code,
        category_full_name = excluded.category_full_name;
    """, (name.strip(), generic_name.strip(), atc_code.strip(), category_full_name.strip(), brand_share))
    conn.commit()
    p_id = cursor.execute("SELECT product_id FROM products WHERE name = ?;", (name.strip(),)).fetchone()[0]
    conn.close()
    return p_id

def log_sale_atomic(product_id: int, sale_date: str, units_requested: float, db_path: Path = DB_PATH) -> Dict[str, Any]:
    """
    Logs a sale atomically:
    1. Finds active batches for the exact medicine, sorted by FEFO (earliest expiry first).
    2. Deducts stock across batches.
    3. If units_requested > total stock, records remaining as unmet_demand.
    4. Updates daily_sales and logs stock_movements ledger.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
        SELECT batch_id, quantity_on_hand, expiry_date 
        FROM inventory_batches 
        WHERE product_id = ? AND quantity_on_hand > 0 AND received_date <= ?
        ORDER BY expiry_date ASC;
        """, (product_id, sale_date))
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
            
            cursor.execute("UPDATE inventory_batches SET quantity_on_hand = ? WHERE batch_id = ?", (new_stock, b_id))
            
            cursor.execute("""
            INSERT INTO stock_movements (batch_id, movement_date, movement_type, quantity_change, notes)
            VALUES (?, ?, 'SALE', ?, 'Daily POS dispensing');
            """, (b_id, sale_date, -deduct))

            units_left_to_fill -= deduct
            actual_units_dispensed += deduct

        unmet = max(0.0, units_left_to_fill)

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
        clean_lot = batch_id.strip()
        cursor.execute("SELECT batch_id FROM inventory_batches WHERE batch_id = ?", (clean_lot,))
        if cursor.fetchone():
            raise ValueError(f"Batch LOT '{clean_lot}' already exists in inventory. Please enter a unique Batch LOT ID.")

        cursor.execute("""
        INSERT INTO inventory_batches (batch_id, product_id, initial_quantity, quantity_on_hand, received_date, expiry_date)
        VALUES (?, ?, ?, ?, ?, ?);
        """, (clean_lot, product_id, float(initial_qty), float(initial_qty), str(received_date), str(expiry_date)))

        cursor.execute("""
        INSERT INTO stock_movements (batch_id, movement_date, movement_type, quantity_change, notes)
        VALUES (?, ?, 'DELIVERY', ?, 'Supplier delivery receipt');
        """, (clean_lot, str(received_date), float(initial_qty)))

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def log_sales_bulk_atomic(sales_list: List[Dict[str, Any]], sale_date: str, db_path: Path = DB_PATH) -> Dict[str, Any]:
    """
    Atomically processes a list of dispensing entries for a given sale date:
    sales_list: [{'product_id': int, 'product_name': str, 'units': float}, ...]
    
    Guarantees ACID transactions:
    All deductions across physical batches and daily_sales updates occur in a SINGLE transaction.
    If any single deduction fails, the entire transaction is rolled back with zero partial state.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    try:
        total_dispensed = 0.0
        total_unmet = 0.0
        processed_items = []
        
        for item in sales_list:
            p_id = int(item['product_id'])
            p_name = item.get('product_name', f'Product #{p_id}')
            units_requested = float(item['units'])
            if units_requested <= 0:
                continue
                
            cursor.execute("""
            SELECT batch_id, quantity_on_hand, expiry_date 
            FROM inventory_batches 
            WHERE product_id = ? AND quantity_on_hand > 0 AND received_date <= ?
            ORDER BY expiry_date ASC;
            """, (p_id, sale_date))
            batches = cursor.fetchall()

            units_left_to_fill = units_requested
            actual_units_dispensed = 0.0
            batches_depleted = []

            for b in batches:
                if units_left_to_fill <= 0:
                    break
                
                b_id = b['batch_id']
                b_stock = float(b['quantity_on_hand'])
                b_exp = b['expiry_date']

                deduct = min(b_stock, units_left_to_fill)
                new_stock = b_stock - deduct
                
                cursor.execute("UPDATE inventory_batches SET quantity_on_hand = ? WHERE batch_id = ?", (new_stock, b_id))
                
                cursor.execute("""
                INSERT INTO stock_movements (batch_id, movement_date, movement_type, quantity_change, notes)
                VALUES (?, ?, 'SALE', ?, 'Daily bulk POS dispensing');
                """, (b_id, sale_date, -deduct))

                units_left_to_fill -= deduct
                actual_units_dispensed += deduct
                batches_depleted.append({
                    'batch_id': b_id,
                    'deducted': deduct,
                    'remaining': new_stock,
                    'expiry_date': b_exp
                })

            unmet = max(0.0, units_left_to_fill)

            cursor.execute("""
            INSERT INTO daily_sales (product_id, sale_date, units_sold, unmet_demand)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(product_id, sale_date) DO UPDATE SET
                units_sold = units_sold + excluded.units_sold,
                unmet_demand = unmet_demand + excluded.unmet_demand;
            """, (p_id, sale_date, actual_units_dispensed, unmet))

            total_dispensed += actual_units_dispensed
            total_unmet += unmet
            processed_items.append({
                'product_id': p_id,
                'name': p_name,
                'requested': units_requested,
                'dispensed': actual_units_dispensed,
                'unmet_demand': unmet,
                'batches_depleted': batches_depleted
            })

        conn.commit()
        return {
            'total_medicines': len(processed_items),
            'total_dispensed': total_dispensed,
            'total_unmet': total_unmet,
            'items': processed_items
        }

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_daily_sales_for_date(sale_date: str, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves all medicines dispensed on a specific calendar date with ATC metadata."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT s.sale_id, s.product_id, p.name, p.generic_name, p.category_full_name, s.units_sold, s.unmet_demand
    FROM daily_sales s
    JOIN products p ON s.product_id = p.product_id
    WHERE s.sale_date = ? AND (s.units_sold > 0 OR s.unmet_demand > 0)
    ORDER BY s.units_sold DESC;
    """, (sale_date,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
