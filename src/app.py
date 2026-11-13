"""
app.py — Modern, Clinical-Grade Decision Support Dashboard for Pharmacies.
Built with Streamlit, custom SaaS styling, and atomic SQLite operations.
Features 38 Essential Indian Household Medicines across 8 ATC Categories.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path

import sys
import importlib

# Ensure fresh import of local modules across Streamlit hot reloads
for mod in ['db', 'engine', 'features']:
    if mod in sys.modules:
        try:
            importlib.reload(sys.modules[mod])
        except Exception:
            pass

import db
if not hasattr(db, 'add_product'):
    try:
        importlib.reload(db)
    except Exception:
        pass

# Resilient function bindings to prevent any runtime ImportError
get_connection = db.get_connection
log_sale_atomic = db.log_sale_atomic
add_batch = db.add_batch
DB_PATH = db.DB_PATH

log_sales_bulk_atomic = getattr(db, 'log_sales_bulk_atomic', None)
if log_sales_bulk_atomic is None:
    importlib.reload(db)
    log_sales_bulk_atomic = db.log_sales_bulk_atomic

get_daily_sales_for_date = getattr(db, 'get_daily_sales_for_date', None)
if get_daily_sales_for_date is None:
    importlib.reload(db)
    get_daily_sales_for_date = db.get_daily_sales_for_date

if hasattr(db, 'add_product'):
    add_product = db.add_product
else:
    def add_product(name: str, generic_name: str, atc_code: str, category_full_name: str, brand_share: float = 0.10, db_path: Path = None) -> int:
        target_path = db_path or DB_PATH
        conn = get_connection(target_path)
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
    db.add_product = add_product

from engine import evaluate_reorder_risk, evaluate_batch_expiry_fefo, MODEL_MAE
from features import prepare_inference_row, FEATURE_COLUMNS, CATEGORY_MAPPING

# Clean, clinically readable category display labels (prevent badge wrapping & overflow)
CATEGORY_DISPLAY_LABELS = {
    'M01AB_Anti_Inflammatory_Acetic': 'Anti-Inflammatory (Acetic)',
    'M01AE_Anti_Inflammatory_Propionic': 'Anti-Inflammatory (Propionic)',
    'N02BA_Analgesics_Salicylic': 'Analgesics (Salicylic)',
    'N02BE_Analgesics_Paracetamol_Anilides': 'Analgesics (Paracetamol)',
    'N05B_Psycholeptics_Anxiolytics': 'Psycholeptics (Anxiolytics)',
    'N05C_Psycholeptics_Hypnotics_Sedatives': 'Hypnotics & Sedatives',
    'R03_Respiratory_Airway_Diseases': 'Respiratory & Airway',
    'R06_Antihistamines_Systemic': 'Antihistamines (Systemic)',
}

# -----------------------------------------------------------------------------
# SAFE HTML RENDERER (ELIMINATES ALL MARKDOWN INDENTED CODE BLOCK GLITCHES)
# -----------------------------------------------------------------------------
def render_html(html_str: str):
    """
    Safely renders HTML without triggering Markdown's 4-space code block parser.
    Strips all leading whitespace from each line and drops blank lines.
    Uses st.html() when available (Streamlit >= 1.34) or st.markdown(unsafe_allow_html=True).
    """
    clean = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
    if hasattr(st, "html"):
        st.html(clean)
    else:
        st.markdown(clean, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PharmaPulse — Indian Dispensary Inventory & Expiry Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling: Forest Green with Starry Dotted Texture & Electric Purple Highlights
RAW_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;600&display=swap');

/* Completely Hide Streamlit Top Header Bar (Eliminates the black cut-off strip) */
header, [data-testid="stHeader"], [data-testid="stAppHeader"], .stAppHeader {
    display: none !important;
    height: 0px !important;
    visibility: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}
#MainMenu {
    visibility: hidden !important;
    display: none !important;
}
footer {
    visibility: hidden !important;
    display: none !important;
}

/* Global Background: Forest Green Gradient + Luminous Dotted Matrix Stipple + Electric Purple Auras */
.stApp {
    background: 
        radial-gradient(rgba(237, 230, 216, 0.14) 1.2px, transparent 1.2px),
        radial-gradient(circle at 82% 16%, rgba(139, 92, 246, 0.22) 0%, transparent 42%),
        radial-gradient(circle at 18% 65%, rgba(124, 58, 237, 0.18) 0%, transparent 46%),
        linear-gradient(180deg, #091c12 0%, #102d1d 22%, #173e2b 45%, #235239 65%, #466853 82%, #d5cbb8 100%) !important;
    background-size: 22px 22px, 100% 100%, 100% 100%, 100% 100% !important;
    background-attachment: fixed !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #f1ede4 !important;
}

/* Container Spacing */
.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1240px !important;
}

/* Headings in Warm Sand & Editorial Serif */
h1, h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #f3ece0 !important;
    letter-spacing: -0.02em !important;
}

/* Brand Title & Subtitle */
.brand-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #fbf7ef;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.15;
}
.brand-subtitle {
    font-size: 0.85rem;
    color: #b5c7b8;
    margin-top: 4px;
    letter-spacing: 0.3px;
}

/* Electric Purple Accents & Glow */
.electric-pill {
    background: rgba(139, 92, 246, 0.22);
    color: #c4b5fd;
    border: 1px solid rgba(168, 85, 247, 0.5);
    box-shadow: 0 0 16px rgba(139, 92, 246, 0.35);
    border-radius: 9999px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    display: inline-block;
    text-transform: uppercase;
}
.spotlight-card {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.22) 0%, rgba(17, 38, 26, 0.82) 65%);
    border: 1px solid rgba(168, 85, 247, 0.45);
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.2);
    border-radius: 18px;
    padding: 18px 24px;
    margin-bottom: 1.4rem;
}

/* Clinical Advisory Notice (Warm Sand Frosted Callout) */
.clinical-banner {
    background: rgba(237, 228, 211, 0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(237, 228, 211, 0.2);
    border-left: 4px solid #ede4d3;
    padding: 13px 18px;
    border-radius: 12px;
    font-size: 0.85rem;
    color: #ece5d8;
    margin: 0 0 1.5rem 0;
    line-height: 1.55;
}

/* Three Big Navigation Panels */
div[data-testid="stColumn"] .stButton > button {
    height: 82px !important;
    white-space: pre-line !important;
    text-align: center !important;
    border-radius: 18px !important;
    font-size: 0.96rem !important;
    padding: 12px 18px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* Primary (Active Navigation Panel): Sand Beach Pill */
button[kind="primary"] {
    background: #ede4d3 !important;
    color: #12281b !important;
    border: 2px solid #ede4d3 !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28) !important;
    font-weight: 700 !important;
}
button[kind="primary"] * {
    color: #12281b !important;
    font-weight: 700 !important;
}

/* Secondary (Inactive Navigation Panel): Frosted Forest Glass */
button[kind="secondary"] {
    background: rgba(17, 38, 26, 0.6) !important;
    color: #d1dacd !important;
    border: 1px solid rgba(237, 230, 216, 0.18) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15) !important;
    font-weight: 500 !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
}
button[kind="secondary"]:hover {
    background: rgba(23, 52, 36, 0.85) !important;
    border-color: rgba(237, 230, 216, 0.45) !important;
    color: #fdfbf7 !important;
    transform: translateY(-2px) !important;
}

/* Top KPI Cards */
.kpi-card {
    padding: 18px 22px;
    border-radius: 16px;
    background: rgba(17, 38, 26, 0.68);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(237, 230, 216, 0.16);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.16);
    transition: transform 0.2s ease, border-color 0.2s ease;
    margin-bottom: 1.2rem;
}
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(237, 230, 216, 0.35);
}
.kpi-label {
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #b5c4b7;
}
.kpi-value {
    font-size: 1.95rem;
    font-weight: 700;
    margin: 8px 0 4px 0;
    font-family: 'Playfair Display', Georgia, serif;
    color: #f7f2e8;
}
.kpi-sub {
    font-size: 0.8rem;
    color: #9cb0a0;
}

/* Decision Triage Cards */
.triage-card {
    background: rgba(17, 38, 26, 0.78);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(237, 230, 216, 0.18);
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 18px;
    box-shadow: 0 8px 26px rgba(0, 0, 0, 0.18);
    transition: all 0.2s ease;
}
.triage-card:hover {
    border-color: rgba(237, 230, 216, 0.38);
    transform: translateY(-2px);
}
.triage-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 14px;
    border-bottom: 1px solid rgba(237, 230, 216, 0.1);
    padding-bottom: 12px;
}
.triage-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #fbf7ef;
    margin: 0;
}
.triage-subtitle {
    font-size: 0.82rem;
    color: #a4b7a7;
    letter-spacing: 0.4px;
    margin-top: 3px;
}

/* Badges */
.badge-critical {
    background-color: rgba(220, 38, 38, 0.22);
    color: #fca5a5;
    padding: 5px 13px;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 600;
    border: 1px solid rgba(239, 68, 68, 0.45);
    display: inline-block;
}
.badge-warning {
    background-color: rgba(217, 119, 6, 0.22);
    color: #fde68a;
    padding: 5px 13px;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 600;
    border: 1px solid rgba(245, 158, 11, 0.45);
    display: inline-block;
}
.badge-optimal {
    background-color: rgba(16, 185, 129, 0.22);
    color: #a7f3d0;
    padding: 5px 13px;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 600;
    border: 1px solid rgba(52, 211, 153, 0.45);
    display: inline-block;
}

/* 4-Metric Grid */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 14px 0;
}
.metric-tile {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(237, 230, 216, 0.12);
    border-radius: 12px;
    padding: 12px 14px;
}
.metric-label {
    font-size: 0.74rem;
    color: #a4b7a7;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}
.metric-val {
    font-size: 1.28rem;
    font-weight: 700;
    color: #f7f2e8;
    margin-top: 4px;
}

/* Stock Runway Visual Meter */
.runway-container {
    margin: 14px 0 16px 0;
    background: rgba(0, 0, 0, 0.25);
    border-radius: 10px;
    padding: 10px 14px;
    border: 1px solid rgba(237, 230, 216, 0.08);
}
.runway-bar-bg {
    height: 8px;
    border-radius: 9999px;
    background: rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
    margin: 6px 0;
}
.runway-bar-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.3s ease;
}

/* Action Protocol Callout */
.action-box {
    background: rgba(237, 228, 211, 0.08);
    border-left: 3px solid #ede4d3;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 12px;
    font-size: 0.86rem;
    color: #ede4d3;
    line-height: 1.5;
}

/* Custom Glass Table for Panel 2 */
.custom-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 8px;
    margin: 12px 0;
}
.custom-table th {
    background: rgba(17, 38, 26, 0.85);
    color: #c5d4c8;
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    padding: 12px 16px;
    border-bottom: 1px solid rgba(237, 230, 216, 0.15);
    text-align: left;
    white-space: nowrap;
}
.custom-table td {
    background: rgba(17, 38, 26, 0.65);
    backdrop-filter: blur(12px);
    padding: 14px 16px;
    font-size: 0.88rem;
    color: #f7f2e8;
    border-top: 1px solid rgba(237, 230, 216, 0.08);
    border-bottom: 1px solid rgba(237, 230, 216, 0.08);
}
.custom-table tr td:first-child {
    border-left: 1px solid rgba(237, 230, 216, 0.08);
    border-top-left-radius: 12px;
    border-bottom-left-radius: 12px;
}
.custom-table tr td:last-child {
    border-right: 1px solid rgba(237, 230, 216, 0.08);
    border-top-right-radius: 12px;
    border-bottom-right-radius: 12px;
}
.custom-table tr:hover td {
    background: rgba(23, 52, 36, 0.85);
    border-color: rgba(237, 230, 216, 0.22);
}
.custom-table th, .custom-table td {
    vertical-align: middle;
}
.cat-pill {
    background: rgba(255, 255, 255, 0.07);
    padding: 4px 11px;
    border-radius: 9999px;
    font-size: 0.74rem;
    font-weight: 500;
    border: 1px solid rgba(237, 230, 216, 0.16);
    color: #d1dacd;
    display: inline-block !important;
    white-space: nowrap !important;
    line-height: 1.4;
    letter-spacing: 0.2px;
}

/* Glass Form Card for Panel 3 */
.pos-glass-card {
    background: rgba(17, 38, 26, 0.72);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(237, 230, 216, 0.18);
    border-radius: 18px;
    padding: 24px 26px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}

/* Input Fields & Date Picker */
div[data-baseweb="input"], div[data-baseweb="select"] {
    background: rgba(17, 38, 26, 0.65) !important;
    border: 1px solid rgba(237, 230, 216, 0.22) !important;
    border-radius: 10px !important;
    color: #f7f2e8 !important;
}
div[data-baseweb="input"] input {
    color: #f7f2e8 !important;
    font-weight: 500 !important;
}
div[data-testid="stDateInput"] label {
    color: #c5d4c8 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px !important;
}

/* Styled Expanders (Collapsible Dropdown Drawers with Arrows) */
div[data-testid="stExpander"] {
    background: rgba(17, 38, 26, 0.72) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(237, 230, 216, 0.18) !important;
    border-radius: 16px !important;
    margin-bottom: 18px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.16) !important;
    transition: all 0.25s ease !important;
}
div[data-testid="stExpander"]:hover {
    border-color: rgba(168, 85, 247, 0.45) !important;
    box-shadow: 0 6px 24px rgba(124, 58, 237, 0.16) !important;
}
div[data-testid="stExpander"] summary {
    padding: 16px 22px !important;
    font-size: 1.08rem !important;
    font-weight: 700 !important;
    color: #fbf7ef !important;
    cursor: pointer !important;
}
div[data-testid="stExpander"] summary svg {
    color: #c4b5fd !important;
    width: 22px !important;
    height: 22px !important;
    transition: transform 0.2s ease !important;
}
div[data-testid="stExpander"] div[role="region"] {
    padding: 18px 24px 26px 24px !important;
    border-top: 1px solid rgba(237, 230, 216, 0.1) !important;
    background: rgba(11, 28, 19, 0.55) !important;
}
</style>
"""
render_html(RAW_CSS)

# -----------------------------------------------------------------------------
# 2. MODEL ARTIFACT LOADER (SILENT & RESILIENT)
# -----------------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent

@st.cache_resource
def get_model():
    """Attempts to load CatBoost artifact silently without breaking UI if uninstalled."""
    for filename in ["pharma_catboost_demand_model.cbm", "pharma_catboost_demand_model.pkl"]:
        p = MODEL_DIR / filename
        if p.exists():
            try:
                if p.suffix == ".cbm":
                    from catboost import CatBoostRegressor
                    m = CatBoostRegressor()
                    m.load_model(str(p))
                    return m
                elif p.suffix == ".pkl":
                    import joblib
                    return joblib.load(p)
            except Exception:
                pass
    return None

model = get_model()

# -----------------------------------------------------------------------------
# 3. HEADER & DATE SELECTOR (CLEAR & FULLY VISIBLE)
# -----------------------------------------------------------------------------
conn = get_connection()
bounds = conn.execute("SELECT MIN(sale_date) AS lo, MAX(sale_date) AS hi FROM daily_sales;").fetchone()
conn.close()
data_min = pd.to_datetime(bounds['lo']).date() if bounds['lo'] else None
data_max = pd.to_datetime(bounds['hi']).date() if bounds['hi'] else None

head_col1, head_col2 = st.columns([2.7, 1.3], vertical_alignment="center")

with head_col1:
    render_html("""
    <div style="display: flex; align-items: center; gap: 16px; padding: 4px 0 10px 0;">
        <span style="font-size: 2.2rem;">🌿</span>
        <div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <h1 class="brand-title">PharmaPulse</h1>
                <span class="electric-pill">✦ Clinical Intelligence</span>
            </div>
            <div class="brand-subtitle">Indian Dispensary Decision Support • 38 Essential Household Formulations</div>
        </div>
    </div>
    """)

with head_col2:
    as_of_date = st.date_input(
        "📅 Simulation Cutoff Date",
        value=data_max or date(2019, 10, 8),
        min_value=data_min,
        max_value=data_max,
        help="Point-in-time cutoff. All forecasting models strictly use sales data recorded up to this day."
    )

# Clean, Prominent Clinical Advisory Guidance Box
render_html("""
<div class="clinical-banner">
    <strong style="color: #f7f2e8; letter-spacing: 0.3px;">⚕️ Clinical Advisory Notice:</strong> 
    All demand projections, stockout alerts, and batch expiry assessments are operational decision-support aids for replenishment planning. 
    A registered pharmacist retains complete statutory authority over all purchasing, dispensing, discounting, and disposal decisions.
</div>
""")

# -----------------------------------------------------------------------------
# 4. DATA RETRIEVAL & REAL-TIME INFERENCE
# -----------------------------------------------------------------------------
conn = get_connection()
products_db = conn.execute("""
    SELECT product_id, name, generic_name, atc_code, category_full_name, brand_share 
    FROM products 
    ORDER BY category_full_name, name;
""").fetchall()

batches_df = pd.read_sql_query("""
    SELECT b.batch_id, b.product_id, p.name as product_name, p.generic_name, p.category_full_name,
           b.quantity_on_hand, b.initial_quantity, b.expiry_date, b.received_date
    FROM inventory_batches b
    JOIN products p ON b.product_id = p.product_id
    WHERE b.quantity_on_hand > 0 AND b.received_date <= ?;
""", conn, params=(str(as_of_date),))

sales_df = pd.read_sql_query("""
    SELECT s.sale_date as datum, p.category_full_name as category, s.units_sold
    FROM daily_sales s
    JOIN products p ON s.product_id = p.product_id;
""", conn)
conn.close()

if not products_db:
    st.error("Database is empty. Please run `python src/seed_db.py` to initialize Indian household medicines and sample inventory.")
    st.stop()

if model is None:
    st.error("Model artifact `pharma_catboost_demand_model.pkl` could not be loaded. Check that `catboost` and `joblib` are installed.")
    st.stop()

if sales_df.empty:
    st.error("No sales history found. Please run `python src/seed_db.py` to load daily sales.")
    st.stop()

# Live 28-day forecasts per ATC category using only sales history up to cutoff
feature_row = prepare_inference_row(sales_df, as_of_date)
if feature_row[FEATURE_COLUMNS].isna().any().any():
    st.error(f"Not enough sales history before {as_of_date} to build features (52 weeks required).")
    st.stop()
cat_forecast_map = dict(zip(feature_row['category'], model.predict(feature_row[FEATURE_COLUMNS])))

# -----------------------------------------------------------------------------
# 5. CORE DECISION EVALUATION FOR EACH MEDICINE
# -----------------------------------------------------------------------------
medicine_triage_items = []
all_catalog_rows = []
total_units_wasting_soon = 0.0
total_units_already_expired = 0.0
reorder_count = 0
expiry_risk_count = 0
optimal_count = 0

for prod in products_db:
    p_id = prod['product_id']
    p_name = prod['name']
    p_generic = prod['generic_name']
    p_atc = prod['atc_code']
    p_cat = prod['category_full_name']
    p_share = float(prod['brand_share'])

    # Category demand apportioned to this specific Indian brand
    category_28d_demand = max(0.0, float(cat_forecast_map[p_cat]))
    medicine_demand_28d = max(0.1, category_28d_demand * p_share)

    # Statistically honest prediction error scaled from verified CatBoost test MAE (35.13 units)
    brand_mae = round(MODEL_MAE * p_share, 1)

    # Stock batches for this specific medicine
    prod_batches = batches_df[batches_df['product_id'] == p_id].to_dict('records') if not batches_df.empty else []
    total_stock = sum(b['quantity_on_hand'] for b in prod_batches)

    # Separate already expired batches from valid active batches
    expired_batches = [b for b in prod_batches if pd.to_datetime(b['expiry_date']) <= pd.to_datetime(as_of_date)]
    valid_batches = [b for b in prod_batches if pd.to_datetime(b['expiry_date']) > pd.to_datetime(as_of_date)]
    valid_batches = sorted(valid_batches, key=lambda b: pd.to_datetime(b['expiry_date']))

    expired_units = sum(b['quantity_on_hand'] for b in expired_batches)
    usable_stock = sum(b['quantity_on_hand'] for b in valid_batches)
    total_units_already_expired += expired_units

    # 1. Evaluate Reorder Need based strictly on USABLE stock (expired stock cannot be dispensed)
    reorder_eval = evaluate_reorder_risk(
        product_name=p_name,
        total_stock_on_hand=usable_stock,
        predicted_28d_demand=medicine_demand_28d,
        lead_time_days=7,
        safety_stock_days=5,
        mae=brand_mae
    )
    demand_lower = reorder_eval['forecast_range'][0]
    demand_upper = reorder_eval['forecast_range'][1]

    # 2. Evaluate Batch Expiry Risk (FEFO) on all batches
    expiry_evals = evaluate_batch_expiry_fefo(
        batches=prod_batches,
        predicted_28d_demand=medicine_demand_28d,
        as_of_date_str=str(as_of_date),
        horizon_days=28
    )

    future_waste = sum(e['expected_waste_units'] for e in expiry_evals if e['expiry_flag'] != 'EXPIRED')
    total_waste = expired_units + future_waste
    prod_waste = total_waste
    total_units_wasting_soon += total_waste

    next_valid_batch = valid_batches[0] if valid_batches else None

    has_expired = (expired_units > 0)
    has_future_waste = (future_waste > 0)
    is_waste = (total_waste > 0)
    is_reorder = (reorder_eval['reorder_flag'] in ['CRITICAL_STOCKOUT', 'REORDER_SOON'])

    if is_waste:
        expiry_risk_count += 1
    if is_reorder:
        reorder_count += 1
    if not is_waste and not is_reorder:
        optimal_count += 1

    actions = []
    if has_expired:
        exp_lots = ", ".join(f"<code>{b['batch_id']}</code>" for b in expired_batches)
        actions.append(
            f"🚫 <strong>QUARANTINE & DISCARD EXPIRED STOCK:</strong> Batch {exp_lots} ({expired_units:.0f} units) "
            f"has reached/passed its expiry date. <u>Immediately remove from dispensing shelves</u> and quarantine for formal disposal or vendor credit (RMA). "
            f"<strong>DO NOT DISPENSE TO PATIENTS.</strong>"
        )
        if next_valid_batch:
            days_next = (pd.to_datetime(next_valid_batch['expiry_date']) - pd.to_datetime(as_of_date)).days
            actions.append(
                f"📦 <strong>Transition to Next Active Batch:</strong> Advance valid batch <code>{next_valid_batch['batch_id']}</code> "
                f"({next_valid_batch['quantity_on_hand']:.0f} units, expires in {days_next} days on {next_valid_batch['expiry_date']}) "
                f"to the active front dispensing counter."
            )
        else:
            actions.append(
                f"🚨 <strong>ZERO USABLE STOCK:</strong> No unexpired stock remains in the dispensary! "
                f"Place an emergency replenishment order for ~{reorder_eval['suggested_reorder_qty']:.0f} units immediately."
            )

    if has_future_waste:
        first_v = valid_batches[0]
        days_v = (pd.to_datetime(first_v['expiry_date']) - pd.to_datetime(as_of_date)).days
        actions.append(
            f"🏷️ <strong>FEFO Shelf Priority:</strong> Prioritize dispensing from batch <code>{first_v['batch_id']}</code> (expires in {days_v}d). "
            f"Approximately ~{future_waste:.0f} units projected to exceed expected demand before expiry. Plan clearance discount or vendor credit."
        )

    if is_reorder and not (has_expired and not next_valid_batch):
        actions.append(
            f"📦 <strong>Replenishment Order:</strong> Place purchase order for ~{reorder_eval['suggested_reorder_qty']:.0f} units today. "
            f"Usable stock covers only {reorder_eval['days_of_inventory']:.1f} days (Supplier lead time: 7 days)."
        )

    if not is_waste and not is_reorder:
        actions.append("✅ <strong>Healthy Stock:</strong> Stock coverage safely covers expected 28-day demand. Standard dispensing.")

    if expired_batches:
        eb = expired_batches[0]
        if next_valid_batch:
            days_next = (pd.to_datetime(next_valid_batch['expiry_date']) - pd.to_datetime(as_of_date)).days
            batch_display_html = f"<span style='color: #ef4444; font-weight: 700;'>🚫 {eb['batch_id']} (EXPIRED)</span><div style='font-size: 0.72rem; color: #a7f3d0; margin-top: 2px;'>Next active: {next_valid_batch['batch_id']} in {days_next}d</div>"
        else:
            batch_display_html = f"<span style='color: #ef4444; font-weight: 700;'>🚫 {eb['batch_id']} (EXPIRED)</span><div style='font-size: 0.72rem; color: #fca5a5; margin-top: 2px;'>⚠️ No valid batches left!</div>"
    elif valid_batches:
        vb = valid_batches[0]
        days_v = (pd.to_datetime(vb['expiry_date']) - pd.to_datetime(as_of_date)).days
        if len(valid_batches) > 1:
            v_next = valid_batches[1]
            days_next = (pd.to_datetime(v_next['expiry_date']) - pd.to_datetime(as_of_date)).days
            batch_display_html = f"<span style='color: {'#fca5a5' if days_v <= 30 else '#a7f3d0'}; font-weight: 600;'>{vb['batch_id']} in {days_v}d</span><div style='font-size: 0.72rem; color: #a4b7a7; margin-top: 2px;'>Next batch: {v_next['batch_id']} in {days_next}d</div>"
        else:
            batch_display_html = f"<span style='color: {'#fca5a5' if days_v <= 30 else '#a7f3d0'}; font-weight: 600;'>{vb['batch_id']} in {days_v}d</span><div style='font-size: 0.72rem; color: #8fa593; margin-top: 2px;'>Single batch in stock</div>"
    else:
        batch_display_html = "<span style='color: #a4b7a7;'>No active batches</span>"

    medicine_triage_items.append({
        "id": p_id,
        "name": p_name,
        "generic": p_generic,
        "atc_code": p_atc,
        "category": p_cat,
        "brand_share": p_share,
        "brand_mae": brand_mae,
        "total_stock": total_stock,
        "usable_stock": usable_stock,
        "expired_units": expired_units,
        "demand_28d": medicine_demand_28d,
        "demand_lower": demand_lower,
        "demand_upper": demand_upper,
        "days_of_inventory": reorder_eval['days_of_inventory'],
        "suggested_reorder_qty": reorder_eval['suggested_reorder_qty'],
        "reorder_flag": reorder_eval['reorder_flag'],
        "reorder_badge": reorder_eval['reorder_badge'],
        "prod_waste": total_waste,
        "expired_batches": expired_batches,
        "valid_batches": valid_batches,
        "has_expired": has_expired,
        "has_future_waste": has_future_waste,
        "is_waste": is_waste,
        "is_reorder": is_reorder,
        "expiry_evals": expiry_evals,
        "batch_display_html": batch_display_html,
        "actions": actions
    })

    cat_friendly = CATEGORY_DISPLAY_LABELS.get(p_cat, p_cat.split('_', 1)[1].replace('_', ' ') if '_' in p_cat else p_cat)
    doi_val = float(reorder_eval['days_of_inventory'])
    if has_expired:
        severity = 1
        cat_status_pill = f"<span class='badge-critical' style='background-color: rgba(220, 38, 38, 0.35); border-color: #ef4444; color: #fca5a5;'>⚫ {expired_units:.0f} Expired</span>"
    elif has_future_waste:
        severity = 2
        cat_status_pill = f"<span class='badge-critical'>🔴 ~{future_waste:.0f} Units Risk</span>"
    elif doi_val < 7.0:
        severity = 3
        cat_status_pill = "<span class='badge-critical' style='background-color: rgba(220, 38, 38, 0.35); border-color: #ef4444; color: #fca5a5;'>🚨 Critical Stockout (<7d)</span>"
    elif doi_val < 12.0:
        severity = 4
        cat_status_pill = "<span class='badge-warning'>🟡 Reorder Warning (<12d)</span>"
    elif usable_stock < medicine_demand_28d:
        severity = 5
        cat_status_pill = "<span class='badge-warning'>🟡 Low Stock vs 28d</span>"
    elif doi_val > 70.0:
        severity = 6
        cat_status_pill = "<span style='background: rgba(59, 130, 246, 0.2); color: #93c5fd; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; border: 1px solid rgba(96, 165, 250, 0.4);'>🔵 Overstocked</span>"
    else:
        severity = 7
        cat_status_pill = "<span class='badge-optimal'>🟢 Balanced</span>"

    all_catalog_rows.append({
        "id": p_id,
        "name": p_name,
        "generic": p_generic,
        "category": cat_friendly,
        "atc_code": p_atc,
        "total_stock": total_stock,
        "usable_stock": usable_stock,
        "expired_units": expired_units,
        "demand_28d": medicine_demand_28d,
        "runway": doi_val,
        "reorder_badge": reorder_eval['reorder_badge'],
        "has_expired": has_expired,
        "has_future_waste": has_future_waste,
        "future_waste": future_waste,
        "severity_order": severity,
        "status_pill": cat_status_pill
    })

# -----------------------------------------------------------------------------
# 6. THREE PROMINENT TOP NAVIGATION PANELS
# -----------------------------------------------------------------------------
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Alerts"

nav_c1, nav_c2, nav_c3 = st.columns(3)

items_needing_action = [i for i in medicine_triage_items if i['is_waste'] or i['is_reorder']]
total_monitored_stock = batches_df['quantity_on_hand'].sum() if not batches_df.empty else 0

with nav_c1:
    is_sel = (st.session_state.active_nav == "Alerts")
    sub_txt = f"🚨 {len(items_needing_action)} Formulations Need Review" if items_needing_action else "🟢 All Medicines Healthy"
    if st.button(
        f"⚡ Clinical Triage & Alerts\n{sub_txt}",
        key="nav_btn_alerts",
        use_container_width=True,
        type="primary" if is_sel else "secondary"
    ):
        st.session_state.active_nav = "Alerts"
        st.rerun()

with nav_c2:
    is_sel = (st.session_state.active_nav == "Inventory")
    if st.button(
        f"📦 Full Inventory & Batches\n{len(products_db)} Medicines • {total_monitored_stock:,.0f} Units Monitored",
        key="nav_btn_inv",
        use_container_width=True,
        type="primary" if is_sel else "secondary"
    ):
        st.session_state.active_nav = "Inventory"
        st.rerun()

with nav_c3:
    is_sel = (st.session_state.active_nav == "POS")
    if st.button(
        "📝 Dispensing Ledger (POS)\nReal-time Atomic FEFO Depletion",
        key="nav_btn_pos",
        use_container_width=True,
        type="primary" if is_sel else "secondary"
    ):
        st.session_state.active_nav = "POS"
        st.rerun()

render_html("<div style='margin-bottom: 1.5rem;'></div>")

# -----------------------------------------------------------------------------
# PANEL 1: CLINICAL TRIAGE & DECISION ALERTS
# -----------------------------------------------------------------------------
if st.session_state.active_nav == "Alerts":
    # 1. EXECUTIVE KPI STRIP
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_html(f"""
        <div class="kpi-card" style="border-left: 4px solid #ef4444;">
            <div class="kpi-label">Impending Expiry Waste</div>
            <div class="kpi-value" style="color: #fca5a5;">{total_units_wasting_soon:,.0f} <span style="font-size: 0.95rem; font-weight: 400; color: #b5c4b7;">units</span></div>
            <div class="kpi-sub">{expiry_risk_count} formulation(s) at expiry risk</div>
        </div>
        """)

    with k2:
        render_html(f"""
        <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
            <div class="kpi-label">Reorder Warnings</div>
            <div class="kpi-value" style="color: #fde68a;">{reorder_count} <span style="font-size: 0.95rem; font-weight: 400; color: #b5c4b7;">medicines</span></div>
            <div class="kpi-sub">Stock below 28-day expected demand</div>
        </div>
        """)

    with k3:
        render_html(f"""
        <div class="kpi-card" style="border-left: 4px solid #10b981;">
            <div class="kpi-label">Optimal Stock</div>
            <div class="kpi-value" style="color: #a7f3d0;">{optimal_count} <span style="font-size: 0.95rem; font-weight: 400; color: #b5c4b7;">medicines</span></div>
            <div class="kpi-sub">Coverage safely balanced</div>
        </div>
        """)

    with k4:
        render_html(f"""
        <div class="kpi-card" style="border-left: 4px solid #a855f7;">
            <div class="kpi-label" style="color: #d8b4fe;">Total Physical Stock</div>
            <div class="kpi-value" style="color: #ede4d3;">{total_monitored_stock:,.0f} <span style="font-size: 0.95rem; font-weight: 400; color: #b5c4b7;">units</span></div>
            <div class="kpi-sub">Across {len(products_db)} Indian household brands</div>
        </div>
        """)

    # 2. SPOTLIGHT HIGHLIGHT CARD (Inspired by the Reference Mock's Center Feature)
    urgent_item = next((i for i in medicine_triage_items if i['has_expired']), None) or (items_needing_action[0] if items_needing_action else None)
    if urgent_item:
        spot_title = urgent_item['name']
        if urgent_item['has_expired']:
            spot_badge_title = "Quarantine Alert"
            eb_id = urgent_item['expired_batches'][0]['batch_id'] if urgent_item['expired_batches'] else 'LOT-EXP'
            next_txt = f"Advance unexpired batch <code>{urgent_item['valid_batches'][0]['batch_id']}</code> ({urgent_item['valid_batches'][0]['quantity_on_hand']:.0f} units) to front dispensing counter." if urgent_item['valid_batches'] else "No valid stock left — place emergency order immediately."
            spot_directive = f"<strong>Immediate Shelf Action:</strong> Physically remove &amp; quarantine batch <code>{eb_id}</code> ({urgent_item['expired_units']:.0f} units). {next_txt}"
            spot_status_html = f"<span style='color: #fca5a5; font-weight: 600;'>🚨 {urgent_item['expired_units']:.0f} Units Expired On Shelf</span>"
        elif urgent_item['is_reorder']:
            spot_badge_title = "Reorder Urgency"
            spot_directive = f"<strong>Replenishment Directive:</strong> Stock coverage covers only {urgent_item['days_of_inventory']:.1f} days (Supplier lead time: 7d). Place purchase order for ~{urgent_item['suggested_reorder_qty']:.0f} units today."
            spot_status_html = f"<span style='color: #fde047; font-weight: 600;'>⚠️ Low Runway: {urgent_item['days_of_inventory']:.1f} Days Coverage</span>"
        else:
            spot_badge_title = "Healthy Inventory"
            spot_directive = "Standard FEFO dispensing protocol active. Stock coverage safely balances 28-day expected demand."
            spot_status_html = "<span style='color: #a7f3d0; font-weight: 600;'>🟢 Optimal Stock Levels</span>"

        render_html(f"""
        <div class="spotlight-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.1rem; color: #c4b5fd;">✦</span>
                    <strong style="color: #f3e8ff; text-transform: uppercase; letter-spacing: 0.8px; font-size: 0.76rem;">Dispensary Spotlight Action</strong>
                </div>
                <span class="electric-pill">{spot_badge_title}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h3 style="margin: 0; font-size: 1.35rem; color: #ffffff;">{spot_title}</h3>
                    <div style="font-size: 0.84rem; color: #e9d5ff; margin-top: 4px;">{urgent_item['generic']} &nbsp;•&nbsp; {spot_status_html}</div>
                </div>
                <div style="display: flex; gap: 20px; align-items: flex-end;">
                    {f'''<div style="text-align: right;">
                        <div style="font-size: 0.72rem; color: #fca5a5; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Expired Stock</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">{urgent_item['expired_units']:,.0f} <span style="font-size: 0.8rem; font-weight: 400; color: #fca5a5;">units</span></div>
                    </div>''' if urgent_item['has_expired'] else ''}
                    <div style="text-align: right;">
                        <div style="font-size: 0.72rem; color: #d8b4fe; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Usable Stock</div>
                        <div style="font-size: 1.6rem; font-weight: 700; color: #fdfbf7;">{urgent_item['usable_stock']:,.0f} <span style="font-size: 0.8rem; font-weight: 400; color: #e9d5ff;">units</span></div>
                    </div>
                </div>
            </div>
            <div style="background: rgba(0, 0, 0, 0.28); border-radius: 12px; padding: 11px 16px; margin-top: 14px; border-left: 3px solid {'#f87171' if urgent_item['has_expired'] else '#a855f7'}; font-size: 0.85rem; color: #fdfbf7; line-height: 1.5;">
                {spot_directive}
            </div>
        </div>
        """)

    # 3. TRIAGE FILTER STRIP
    if "triage_filter" not in st.session_state:
        st.session_state.triage_filter = "Action Required"

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        if st.button(f"🚨 Action Required ({len(items_needing_action)})", key="flt_act", use_container_width=True,
                     type="primary" if st.session_state.triage_filter == "Action Required" else "secondary"):
            st.session_state.triage_filter = "Action Required"
            st.rerun()
    with f_col2:
        if st.button(f"🔴 Expiry Risks ({expiry_risk_count})", key="flt_exp", use_container_width=True,
                     type="primary" if st.session_state.triage_filter == "Expiry Risks" else "secondary"):
            st.session_state.triage_filter = "Expiry Risks"
            st.rerun()
    with f_col3:
        if st.button(f"🟡 Low Stock ({reorder_count})", key="flt_low", use_container_width=True,
                     type="primary" if st.session_state.triage_filter == "Low Stock" else "secondary"):
            st.session_state.triage_filter = "Low Stock"
            st.rerun()
    with f_col4:
        if st.button(f"🔘 All {len(products_db)} Medicines", key="flt_all", use_container_width=True,
                     type="primary" if st.session_state.triage_filter == "All" else "secondary"):
            st.session_state.triage_filter = "All"
            st.rerun()

    if st.session_state.triage_filter == "Action Required":
        filtered_items = items_needing_action
    elif st.session_state.triage_filter == "Expiry Risks":
        filtered_items = [i for i in medicine_triage_items if i['is_waste']]
    elif st.session_state.triage_filter == "Low Stock":
        filtered_items = [i for i in medicine_triage_items if i['is_reorder']]
    else:
        filtered_items = medicine_triage_items

    render_html("<div style='margin-bottom: 0.8rem;'></div>")

    # 4. CLINICAL DECISION TRIAGE CARDS
    if filtered_items:
        for item in filtered_items:
            if item['is_waste']:
                border_color = "#ef4444"
            elif item['is_reorder']:
                border_color = "#f59e0b"
            else:
                border_color = "#10b981"

            badges_html = ""
            if item['has_expired']:
                badges_html += f'<span class="badge-critical" style="background-color: rgba(220, 38, 38, 0.35); border-color: #ef4444; color: #fca5a5;">⚫ {item["expired_units"]:.0f} Units Expired (Discard)</span> '
            if item['has_future_waste']:
                future_qty = item['prod_waste'] - item['expired_units']
                badges_html += f'<span class="badge-critical">🔴 ~{future_qty:.0f} Units Expiry Risk</span> '
            if item['is_reorder']:
                badges_html += f'<span class="badge-warning">{item["reorder_badge"]}</span> '
            if not item['is_waste'] and not item['is_reorder']:
                badges_html += '<span class="badge-optimal">🟢 Balanced Coverage</span>'

            doi = item['days_of_inventory']
            runway_pct = min(100, max(0, int((doi / 35.0) * 100)))
            fill_color = "#ef4444" if doi < 7.0 else "#f59e0b" if doi < 12.0 else "#10b981"

            if item['has_expired']:
                stock_sub = f"<div style='font-size: 0.72rem; color: #ef4444; margin-top: 2px;'>⚠️ {item['expired_units']:,.0f} units expired (quarantine)</div>"
                stock_disp_val = f"{item['usable_stock']:,.0f} <span style='font-size: 0.8rem; font-weight: 400; color: #a4b7a7;'>usable</span>"
            else:
                stock_sub = f"<div style='font-size: 0.72rem; color: #8fa593; margin-top: 2px;'>100% usable inventory</div>"
                stock_disp_val = f"{item['total_stock']:,.0f} <span style='font-size: 0.8rem; font-weight: 400; color: #a4b7a7;'>units</span>"

            actions_html = "".join([f"<li style='margin-bottom: 6px;'>{act}</li>" for act in item['actions']])

            card_html = f"""
            <div class="triage-card" style="border-left: 5px solid {border_color};">
                <div class="triage-header">
                    <div>
                        <h3 class="triage-title">{item['name']}</h3>
                        <div class="triage-subtitle">Generic: <strong>{item['generic']}</strong> &nbsp;•&nbsp; ATC: {item['atc_code']} ({CATEGORY_DISPLAY_LABELS.get(item['category'], item['category'])})</div>
                    </div>
                    <div>{badges_html}</div>
                </div>
                <div class="metric-grid">
                    <div class="metric-tile">
                        <div class="metric-label">Stock On Hand</div>
                        <div class="metric-val">{stock_disp_val}</div>
                        {stock_sub}
                    </div>
                    <div class="metric-tile">
                        <div class="metric-label">28-Day Demand Forecast</div>
                        <div class="metric-val">~{item['demand_28d']:,.0f} <span style="font-size: 0.8rem; font-weight: 400; color: #a4b7a7;">units</span></div>
                        <div style="font-size: 0.72rem; color: #a4b7a7; margin-top: 2px;">Test MAE: ±{item['brand_mae']:.1f}u &nbsp;(Range: {item['demand_lower']:.0f}–{item['demand_upper']:.0f})</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-label">Stock Runway</div>
                        <div class="metric-val">{item['days_of_inventory']:.1f} <span style="font-size: 0.8rem; font-weight: 400; color: #a4b7a7;">days</span></div>
                        <div style="font-size: 0.72rem; color: #8fa593; margin-top: 2px;">Lead time: 7d &nbsp;|&nbsp; Safety: 5d</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-label">Earliest Expiring Batch</div>
                        <div class="metric-val" style="font-size: 0.95rem; margin-top: 4px;">{item['batch_display_html']}</div>
                    </div>
                </div>
                <div class="runway-container">
                    <div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: #b5c7b8; margin-bottom: 3px;">
                        <span>0d (Stockout)</span>
                        <span style="color: #f59e0b;">Lead Time (7d)</span>
                        <span style="color: #fde68a;">Safety (12d)</span>
                        <span style="color: #a7f3d0;">28d Target Coverage</span>
                    </div>
                    <div class="runway-bar-bg">
                        <div class="runway-bar-fill" style="width: {runway_pct}%; background: {fill_color};"></div>
                    </div>
                </div>
                <div class="action-box">
                    <strong style="color: #f7f2e8; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.5px;">Recommended Pharmacist Action:</strong>
                    <ul style="margin: 6px 0 0 0; padding-left: 18px;">
                        {actions_html}
                    </ul>
                </div>
            </div>
            """
            render_html(card_html)

            # Progressive Disclosure: FEFO Batch Breakdown Drawer
            if item['expiry_evals']:
                with st.expander(f"🔍 Inspect Physical Batches & FEFO Depletion Breakdown ({item['name']})"):
                    df_b_view = pd.DataFrame(item['expiry_evals'])[[
                        'batch_id', 'expiry_date', 'days_to_expiry', 'current_stock', 'expected_waste_units', 'expiry_badge'
                    ]]
                    st.dataframe(
                        df_b_view, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "batch_id": "Batch LOT",
                            "expiry_date": "Expiration Date",
                            "days_to_expiry": st.column_config.NumberColumn("Days Left", format="%d days"),
                            "current_stock": st.column_config.NumberColumn("Units In Batch", format="%d"),
                            "expected_waste_units": st.column_config.NumberColumn("Projected Waste", format="%.0f units"),
                            "expiry_badge": "FEFO Status"
                        }
                    )
    else:
        st.success("🎉 No medicines match the selected filter. Inventory is healthy and properly balanced!")

    st.markdown("---")

    with st.expander("🔍 AI Forecasting Drivers & Feature Importance (CatBoost)"):
        st.write("Feature importance computed directly from the validated CatBoost model across historical dispensary sales:")
        importances = model.get_feature_importance()
        drivers = pd.DataFrame({
            "Predictor Feature": FEATURE_COLUMNS,
            "Model Weight (%)": (importances / importances.sum() * 100).round(1)
        }).sort_values("Model Weight (%)", ascending=False)
        st.dataframe(drivers, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# PANEL 2: FULL INVENTORY & BATCH MANAGEMENT (TWO DEDICATED VENUES)
# -----------------------------------------------------------------------------
elif st.session_state.active_nav == "Inventory":
    if "inventory_view_venue" not in st.session_state:
        st.session_state.inventory_view_venue = "catalog"

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        is_sel_cat = (st.session_state.inventory_view_venue == "catalog")
        if st.button(
            f"📋 Medicine Formulation Catalog\n{len(products_db)} Formulations • Stock & Runway Overview",
            key="btn_inv_catalog",
            use_container_width=True,
            type="primary" if is_sel_cat else "secondary"
        ):
            st.session_state.inventory_view_venue = "catalog"
            st.rerun()

    with v_col2:
        is_sel_batch = (st.session_state.inventory_view_venue == "batches")
        if st.button(
            f"📦 Physical Batch Depletion Ledger\n{len(batches_df)} Active Lots Monitored in FEFO Sequence",
            key="btn_inv_batches",
            use_container_width=True,
            type="primary" if is_sel_batch else "secondary"
        ):
            st.session_state.inventory_view_venue = "batches"
            st.rerun()

    render_html("<div style='margin-bottom: 1.4rem;'></div>")

    # VENUE A: MEDICINE FORMULATION CATALOG
    if st.session_state.inventory_view_venue == "catalog":
        st.markdown("### 📋 Complete Indian Dispensary Inventory Catalog")
        st.caption("Real-time physical stock counts, 28-day CatBoost demand forecasts, and stock coverage.")

        # Search & Category Filters
        i_col1, i_col2 = st.columns([2, 1])
        search_q = i_col1.text_input("🔍 Search Medicine or Generic Formulation", placeholder="e.g., Dolo, Paracetamol, Combiflam, Asthalin...").strip().lower()
        
        cat_options = ["All Therapeutic Categories"] + sorted(list(set(r['category'] for r in all_catalog_rows)))
        sel_cat = i_col2.selectbox("Filter by Category", options=cat_options)

        # Filter rows
        displayed_catalog = all_catalog_rows
        if sel_cat != "All Therapeutic Categories":
            displayed_catalog = [r for r in displayed_catalog if r['category'] == sel_cat]
        if search_q:
            displayed_catalog = [r for r in displayed_catalog if search_q in r['name'].lower() or search_q in r['generic'].lower()]

        # Sort catalog so RED and critical risks are at the TOP, yellow warnings in middle, and green balanced at the bottom!
        displayed_catalog = sorted(displayed_catalog, key=lambda x: (x['severity_order'], x['runway'], x['name']))

        # Render Custom Glass Table
        table_rows_html = ""
        for r in displayed_catalog:
            status_pill = r['status_pill']
            runway_color = "#ef4444" if r['runway'] < 7.0 else "#f59e0b" if r['runway'] < 12.0 else "#10b981"

            table_rows_html += f"""
            <tr>
                <td style="vertical-align: middle;">
                    <div style="font-weight: 700; color: #fdfbf7; font-size: 0.96rem; line-height: 1.35; margin-bottom: 3px;">{r['name']}</div>
                    <div style="font-size: 0.76rem; color: #a4b7a7; line-height: 1.35; word-break: break-word;">{r['generic']}</div>
                </td>
                <td style="vertical-align: middle;"><span class="cat-pill">{r['category']}</span></td>
                <td style="text-align: right; vertical-align: middle;">
                    <strong style="color: #fdfbf7; font-size: 1rem;">{r['usable_stock']:,.0f}</strong>
                    <span style="font-size: 0.76rem; color: #a4b7a7; margin-left: 2px;">units</span>
                </td>
                <td style="text-align: right; vertical-align: middle;">
                    <span style="font-size: 0.98rem; font-weight: 600; color: #e2e8f0;">~{r['demand_28d']:.0f}</span>
                    <span style="font-size: 0.76rem; color: #a4b7a7; margin-left: 2px;">units</span>
                </td>
                <td style="text-align: center; vertical-align: middle;"><span style="color: {runway_color}; font-weight: 700; font-size: 0.95rem;">{r['runway']:.1f} d</span></td>
                <td style="text-align: center; vertical-align: middle;">{status_pill}</td>
            </tr>
            """

        custom_inventory_table_html = f"""
        <table class="custom-table">
            <thead>
                <tr>
                    <th style="width: 32%; min-width: 250px;">Medicine & Formulation</th>
                    <th style="width: 20%; min-width: 175px;">Therapeutic Category</th>
                    <th style="width: 12%; min-width: 90px; text-align: right;">Usable Stock</th>
                    <th style="width: 12%; min-width: 95px; text-align: right;">28d Forecast</th>
                    <th style="width: 11%; min-width: 90px; text-align: center;">Stock Runway</th>
                    <th style="width: 13%; min-width: 120px; text-align: center;">Dispensary Status</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html if table_rows_html else "<tr><td colspan='6' style='text-align: center; color: #a4b7a7; padding: 24px;'>No medicines match your search criteria.</td></tr>"}
            </tbody>
        </table>
        """
        render_html(custom_inventory_table_html)

    # VENUE B: PHYSICAL BATCH DEPLETION LEDGER
    else:
        st.markdown("### 📦 Active Physical Batch Depletion Ledger (FEFO Sequence)")
        st.caption("All physical inventory batches sorted in First-Expiring, First-Out sequence with remaining shelf life.")

        b_f1, b_f2 = st.columns([2, 1])
        b_search = b_f1.text_input("🔍 Search Batch LOT ID or Medicine Name", placeholder="e.g., LOT-DYN, Crocin, Combiflam...").strip().lower()

        status_opts = ["All Active Batches", "Expired (Quarantine)", "Critical Expiry (< 30d)", "Approaching Expiry (< 90d)", "Safe (> 90d)"]
        sel_b_status = b_f2.selectbox("Filter by Expiry Urgency", options=status_opts)

        batches_view = batches_df.copy().sort_values('expiry_date')
        if b_search:
            batches_view = batches_view[
                batches_view['batch_id'].str.lower().str.contains(b_search) | 
                batches_view['product_name'].str.lower().str.contains(b_search) |
                batches_view['generic_name'].str.lower().str.contains(b_search)
            ]

        batch_rows_html = ""
        for _, b in batches_view.iterrows():
            exp_dt = pd.to_datetime(b['expiry_date'])
            days_rem = (exp_dt - pd.to_datetime(as_of_date)).days

            if days_rem <= 0:
                batch_cat = "Expired (Quarantine)"
                exp_badge = "<span class='badge-critical' style='background-color: rgba(220, 38, 38, 0.35); border-color: #ef4444;'>⚫ EXPIRED</span>"
                days_str = f"<span style='color: #ef4444; font-weight: 700;'>{days_rem} days (Expired)</span>"
            elif days_rem <= 30:
                batch_cat = "Critical Expiry (< 30d)"
                exp_badge = "<span class='badge-critical'>🔴 Critical Expiry Risk</span>"
                days_str = f"<span style='color: #fca5a5; font-weight: 600;'>{days_rem} days left</span>"
            elif days_rem <= 90:
                batch_cat = "Approaching Expiry (< 90d)"
                exp_badge = "<span class='badge-warning'>🟡 Approaching Expiry</span>"
                days_str = f"<span style='color: #fde68a;'>{days_rem} days left</span>"
            else:
                batch_cat = "Safe (> 90d)"
                exp_badge = "<span class='badge-optimal'>🟢 Safe Batch</span>"
                days_str = f"<span style='color: #a7f3d0;'>{days_rem} days left</span>"

            if sel_b_status != "All Active Batches" and sel_b_status != batch_cat:
                continue

            batch_rows_html += f"""
            <tr>
                <td style="vertical-align: middle;"><code style="background: rgba(0, 0, 0, 0.35); padding: 4px 8px; border-radius: 6px; color: #ede4d3; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; border: 1px solid rgba(237, 230, 216, 0.1);">{b['batch_id']}</code></td>
                <td style="vertical-align: middle;">
                    <div style="font-weight: 700; color: #fdfbf7; font-size: 0.95rem; line-height: 1.35; margin-bottom: 2px;">{b['product_name']}</div>
                    <div style="font-size: 0.76rem; color: #a4b7a7; line-height: 1.3;">{b['generic_name']}</div>
                </td>
                <td style="text-align: right; vertical-align: middle;">
                    <strong style="color: #fdfbf7; font-size: 0.95rem;">{b['quantity_on_hand']:,.0f}</strong>
                    <span style="font-size: 0.78rem; color: #a4b7a7;"> / {b['initial_quantity']:,.0f} u</span>
                </td>
                <td style="text-align: center; vertical-align: middle; font-size: 0.82rem; color: #c5d4c8;">{b['received_date']}</td>
                <td style="text-align: center; vertical-align: middle; font-size: 0.82rem; color: #c5d4c8;">{b['expiry_date']}</td>
                <td style="text-align: center; vertical-align: middle;">{days_str}</td>
                <td style="text-align: center; vertical-align: middle;">{exp_badge}</td>
            </tr>
            """

        batches_table_html = f"""
        <table class="custom-table">
            <thead>
                <tr>
                    <th style="width: 14%; min-width: 115px;">Batch LOT</th>
                    <th style="width: 30%; min-width: 240px;">Medicine Formulation</th>
                    <th style="width: 14%; min-width: 110px; text-align: right;">Units Remaining</th>
                    <th style="width: 10%; min-width: 95px; text-align: center;">Received</th>
                    <th style="width: 10%; min-width: 95px; text-align: center;">Expiry Date</th>
                    <th style="width: 10%; min-width: 95px; text-align: center;">Shelf Life</th>
                    <th style="width: 12%; min-width: 110px; text-align: center;">FEFO Status</th>
                </tr>
            </thead>
            <tbody>
                {batch_rows_html if batch_rows_html else "<tr><td colspan='7' style='text-align: center; color: #a4b7a7; padding: 24px;'>No batches match your filter criteria.</td></tr>"}
            </tbody>
        </table>
        """
        render_html(batches_table_html)

# -----------------------------------------------------------------------------
# PANEL 3: DISPENSING LEDGER (POS) & INVENTORY OPERATIONS (3 COLLAPSIBLE DRAWERS)
# -----------------------------------------------------------------------------
elif st.session_state.active_nav == "POS":
    st.markdown("### 📋 Dispensary Operations & Inventory Counter")
    st.caption("Point-of-sale patient dispensing, new batch receipts for existing stock, and formulary brand registration.")

    if "pos_active_drawer" not in st.session_state:
        st.session_state.pos_active_drawer = "dispense"

    # Top Selector Quick-Pill Navigation
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        is_d = (st.session_state.pos_active_drawer == "dispense")
        if st.button("📝 1. Daily Dispense Counter\nPatient POS & Atomic FEFO Depletion", key="btn_pos_disp",
                     use_container_width=True, type="primary" if is_d else "secondary"):
            st.session_state.pos_active_drawer = "dispense"
            st.rerun()

    with q_col2:
        is_b = (st.session_state.pos_active_drawer == "batch")
        if st.button("📦 2. Receive Batch (Existing Medicine)\nStock Replenishment Delivery", key="btn_pos_batch",
                     use_container_width=True, type="primary" if is_b else "secondary"):
            st.session_state.pos_active_drawer = "batch"
            st.rerun()

    with q_col3:
        is_m = (st.session_state.pos_active_drawer == "medicine")
        if st.button("➕ 3. Register Brand New Medicine\nAdd New Formulation & Category", key="btn_pos_med",
                     use_container_width=True, type="primary" if is_m else "secondary"):
            st.session_state.pos_active_drawer = "medicine"
            st.rerun()

    render_html("<div style='margin-bottom: 1.5rem;'></div>")

    # -------------------------------------------------------------------------
    # VENUE 1: DAILY DISPENSE COUNTER (BULK DAY SHEET)
    # -------------------------------------------------------------------------
    if st.session_state.pos_active_drawer == "dispense":
        render_html("""
        <div class="pos-glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="margin: 0; font-size: 1.35rem; color: #fbf7ef;">📝 Operation 1: Daily Dispense Counter (Bulk Day Sheet)</h3>
                <span class="electric-pill">Comprehensive Formulary</span>
            </div>
            <div style="font-size: 0.85rem; color: #b5c7b8; line-height: 1.5;">
                Record your dispensary's complete day-end patient sales in one unified transaction. Review all 38 essential Indian household medicines grouped by therapeutic class, input the units sold today, and commit everything together. Physical stock is deducted atomically in First-Expiring, First-Out (FEFO) sequence across active batches.
            </div>
        </div>
        """)

        # 1. Date Selector & Day-End Status Overview
        d_c1, d_c2 = st.columns([1, 2])
        with d_c1:
            sale_date_val = st.date_input(
                "📅 Dispensing Calendar Date",
                value=as_of_date,
                key="bulk_pos_sale_date"
            )

        # Retrieve any sales already logged for this exact date
        day_sales_logged = get_daily_sales_for_date(str(sale_date_val))
        total_already_dispensed = sum(s['units_sold'] for s in day_sales_logged)

        with d_c2:
            if day_sales_logged:
                render_html(f"""
                <div style="background: rgba(168, 85, 247, 0.14); border: 1px solid rgba(168, 85, 247, 0.38); border-radius: 12px; padding: 10px 16px; margin-top: 4px;">
                    <div style="font-size: 0.74rem; color: #d8b4fe; font-weight: 700; text-transform: uppercase;">Logged History for {sale_date_val}</div>
                    <div style="font-size: 0.95rem; color: #ffffff; font-weight: 600; margin-top: 2px;">
                        <strong>{total_already_dispensed:,.0f} units</strong> already recorded across <strong>{len(day_sales_logged)} medicine(s)</strong>.
                    </div>
                    <div style="font-size: 0.74rem; color: #c5d4c8; margin-top: 2px;">Submitting below will additively update this day's sales and deplete active FEFO batches.</div>
                </div>
                """)
            else:
                render_html(f"""
                <div style="background: rgba(0, 0, 0, 0.25); border: 1px solid rgba(237, 230, 216, 0.12); border-radius: 12px; padding: 10px 16px; margin-top: 4px;">
                    <div style="font-size: 0.74rem; color: #a4b7a7; font-weight: 700; text-transform: uppercase;">Status for {sale_date_val}</div>
                    <div style="font-size: 0.95rem; color: #ffffff; font-weight: 600; margin-top: 2px;">
                        No patient dispensing recorded yet for this date.
                    </div>
                    <div style="font-size: 0.74rem; color: #8fa593; margin-top: 2px;">Fill in the units sold below and submit the complete sheet together.</div>
                </div>
                """)

        # 2. Transaction Receipt Audit Card (if just committed)
        if "last_bulk_receipt" in st.session_state and st.session_state["last_bulk_receipt"]:
            receipt = st.session_state["last_bulk_receipt"]
            items_list_html = ""
            for it in receipt['items']:
                batch_parts = []
                for b in it['batches_depleted']:
                    rem_color = "#ef4444" if b['remaining'] == 0 else "#a7f3d0"
                    batch_parts.append(
                        f"<code>{b['batch_id']}</code> (-{b['deducted']:.0f}u, <span style='color: {rem_color};'>{b['remaining']:.0f}u left</span>)"
                    )
                batch_str = ", ".join(batch_parts) if batch_parts else "No batches depleted"
                unmet_alert = f" <span style='color: #ef4444; font-weight: 700;'>(⚠️ {it['unmet_demand']:.0f} units unmet stockout)</span>" if it['unmet_demand'] > 0 else ""
                items_list_html += f"""
                <li style="margin-bottom: 7px; font-size: 0.88rem; color: #fdfbf7;">
                    <strong>{it['name']}</strong>: <span style="color: #a7f3d0; font-weight: 700;">{it['dispensed']:,.0f} units</span> dispensed {unmet_alert}
                    <div style="font-size: 0.76rem; color: #d8b4fe; margin-top: 2px;">FEFO Allocations: {batch_str}</div>
                </li>
                """

            render_html(f"""
            <div style="background: rgba(88, 28, 135, 0.28); border: 1.5px solid #a855f7; border-radius: 14px; padding: 18px 20px; margin: 16px 0 24px 0; box-shadow: 0 8px 25px rgba(168, 85, 247, 0.2);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.1rem; color: #c4b5fd;">✦</span>
                        <strong style="color: #f3e8ff; text-transform: uppercase; letter-spacing: 0.8px; font-size: 0.78rem;">Dispensing Transaction Committed (Atomic FEFO)</strong>
                    </div>
                    <span class="electric-pill">{receipt.get('date', str(sale_date_val))}</span>
                </div>
                <div style="font-size: 1.15rem; color: #ffffff; font-weight: 700;">
                    ✅ Successfully recorded {receipt['total_dispensed']:,.0f} units across {receipt['total_medicines']} medicine(s)
                </div>
                <div style="margin-top: 12px; background: rgba(0, 0, 0, 0.3); border-radius: 10px; padding: 12px 16px;">
                    <div style="font-size: 0.74rem; color: #d8b4fe; text-transform: uppercase; font-weight: 700; margin-bottom: 6px;">Itemized Batch Depletion Audit</div>
                    <ul style="margin: 0; padding-left: 18px;">
                        {items_list_html}
                    </ul>
                </div>
            </div>
            """)
            if st.button("✕ Dismiss Receipt", key="btn_dismiss_bulk_receipt"):
                st.session_state["last_bulk_receipt"] = None
                st.rerun()

        # 3. The Unified Daily Dispensing Sheet Form
        with st.form("bulk_daily_dispense_form"):
            render_html("""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: rgba(17, 38, 26, 0.95); border-radius: 8px; border-bottom: 2px solid rgba(237, 230, 216, 0.15); margin-bottom: 12px;">
                <span style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: #c5d4c8; letter-spacing: 0.5px;">Medicine Formulation</span>
                <span style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: #c5d4c8; letter-spacing: 0.5px; margin-right: 180px;">Shelf Stock & FEFO Batch</span>
                <span style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: #c5d4c8; letter-spacing: 0.5px;">Units Dispensed</span>
            </div>
            """)

            # Group medicines by category in stable order
            categories_seen = []
            for p in products_db:
                cat = p['category_full_name']
                if cat not in categories_seen:
                    categories_seen.append(cat)

            qty_inputs = {}

            for cat_code in categories_seen:
                cat_prods = [p for p in products_db if p['category_full_name'] == cat_code]
                cat_name = CATEGORY_DISPLAY_LABELS.get(cat_code, cat_code.split('_', 1)[1].replace('_', ' ') if '_' in cat_code else cat_code)

                render_html(f"""
                <div style="background: rgba(17, 38, 26, 0.85); border-left: 4px solid #a855f7; padding: 8px 14px; border-radius: 8px; margin: 18px 0 10px 0; display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: #fdfbf7; font-size: 0.94rem;">{cat_name}</strong>
                    <span class="cat-pill">{len(cat_prods)} formulations</span>
                </div>
                """)

                for p in cat_prods:
                    p_id = p['product_id']
                    # Look up live batches for this medicine relative to sale_date_val
                    p_batches = batches_df[batches_df['product_id'] == p_id].to_dict('records') if not batches_df.empty else []
                    valid_b = [
                        b for b in p_batches 
                        if pd.to_datetime(b['expiry_date']) > pd.to_datetime(sale_date_val)
                        and pd.to_datetime(b['received_date']) <= pd.to_datetime(sale_date_val)
                        and b['quantity_on_hand'] > 0
                    ]
                    valid_b = sorted(valid_b, key=lambda b: pd.to_datetime(b['expiry_date']))
                    usable_stock = sum(b['quantity_on_hand'] for b in valid_b)

                    if valid_b:
                        front_b = valid_b[0]
                        days_left = (pd.to_datetime(front_b['expiry_date']) - pd.to_datetime(sale_date_val)).days
                        lot_color = "#fca5a5" if days_left <= 30 else "#d8b4fe"
                        stock_info_html = f"""
                        <div style="color: #a7f3d0; font-weight: 600; font-size: 0.88rem;">{usable_stock:,.0f} units usable</div>
                        <div style="font-size: 0.74rem; color: #a4b7a7; margin-top: 2px;">Front: <code style="color: {lot_color};">{front_b['batch_id']}</code> ({days_left}d left)</div>
                        """
                    else:
                        stock_info_html = """
                        <div style="color: #ef4444; font-weight: 700; font-size: 0.88rem;">0 units usable</div>
                        <div style="font-size: 0.74rem; color: #fca5a5; margin-top: 2px;">⚠️ Stockout risk</div>
                        """

                    row_c1, row_c2, row_c3 = st.columns([3, 2, 1.2])

                    with row_c1:
                        render_html(f"""
                        <div style="padding: 4px 0;">
                            <div style="font-weight: 700; color: #fdfbf7; font-size: 0.94rem; line-height: 1.3;">{p['name']}</div>
                            <div style="font-size: 0.76rem; color: #a4b7a7; line-height: 1.3;">{p['generic_name']}</div>
                        </div>
                        """)

                    with row_c2:
                        render_html(f"""
                        <div style="padding: 4px 0;">
                            {stock_info_html}
                        </div>
                        """)

                    with row_c3:
                        qty_inputs[p_id] = st.number_input(
                            f"Units Sold - {p['name']}",
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            key=f"bulk_sale_qty_{p_id}",
                            label_visibility="collapsed"
                        )

            render_html("<div style='margin-top: 20px;'></div>")
            submit_all = st.form_submit_button(
                "⚡ Commit & Record All Dispensed Medicines for This Day",
                use_container_width=True,
                type="primary"
            )

            if submit_all:
                sales_to_record = []
                for p in products_db:
                    p_id = p['product_id']
                    val = float(qty_inputs.get(p_id, 0.0))
                    if val > 0.0:
                        sales_to_record.append({
                            'product_id': p_id,
                            'product_name': p['name'],
                            'units': val
                        })

                if not sales_to_record:
                    st.warning("⚠️ No dispensing quantities were entered (all medicines have 0 units). Please enter the units sold for at least one medicine before submitting.")
                else:
                    try:
                        receipt = log_sales_bulk_atomic(sales_to_record, sale_date=str(sale_date_val))
                        receipt['date'] = str(sale_date_val)
                        st.session_state["last_bulk_receipt"] = receipt
                        st.success(f"✅ Successfully recorded {receipt['total_dispensed']:,.0f} units across {receipt['total_medicines']} medicine(s) for {sale_date_val}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error logging day's sales: {e}")

        # 4. Cumulative dispensing ledger for this day
        if day_sales_logged:
            render_html("<div style='margin-top: 28px;'></div>")
            with st.expander(f"📋 View All Dispensing Already Recorded for {sale_date_val} ({len(day_sales_logged)} medicines, {total_already_dispensed:,.0f} units)", expanded=False):
                logged_table_rows = ""
                for s in day_sales_logged:
                    c_clean = CATEGORY_DISPLAY_LABELS.get(s['category_full_name'], s['category_full_name'].split('_', 1)[1].replace('_', ' ') if '_' in s['category_full_name'] else s['category_full_name'])
                    unmet_badge = f"<span class='badge-critical' style='padding: 2px 8px; font-size: 0.72rem;'>⚠️ {s['unmet_demand']:.0f} unmet</span>" if s['unmet_demand'] > 0 else "<span style='color: #a7f3d0;'>Filled</span>"
                    logged_table_rows += f"""
                    <tr>
                        <td style="vertical-align: middle;">
                            <strong style="color: #fdfbf7;">{s['name']}</strong>
                            <div style="font-size: 0.76rem; color: #a4b7a7;">{s['generic_name']}</div>
                        </td>
                        <td style="vertical-align: middle;"><span class="cat-pill">{c_clean}</span></td>
                        <td style="text-align: right; vertical-align: middle;"><strong style="color: #a7f3d0; font-size: 1rem;">{s['units_sold']:,.0f}</strong> <span style="font-size: 0.76rem; color: #a4b7a7;">units</span></td>
                        <td style="text-align: center; vertical-align: middle;">{unmet_badge}</td>
                    </tr>
                    """

                render_html(f"""
                <table class="custom-table">
                    <thead>
                        <tr>
                            <th style="width: 40%;">Medicine Formulation</th>
                            <th style="width: 25%;">Category</th>
                            <th style="width: 18%; text-align: right;">Total Dispensed</th>
                            <th style="width: 17%; text-align: center;">Fulfillment</th>
                        </tr>
                    </thead>
                    <tbody>
                        {logged_table_rows}
                    </tbody>
                </table>
                """)

    # -------------------------------------------------------------------------
    # VENUE 2: RECEIVE NEW BATCH FOR EXISTING MEDICINE
    # -------------------------------------------------------------------------
    elif st.session_state.pos_active_drawer == "batch":
        render_html("""
        <div class="pos-glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="margin: 0; font-size: 1.35rem; color: #fbf7ef;">📦 Operation 2: Receive New Batch (Existing Medicine)</h3>
                <span style="background: rgba(16, 185, 129, 0.2); color: #a7f3d0; border: 1px solid rgba(52, 211, 153, 0.4); border-radius: 9999px; padding: 4px 12px; font-size: 0.72rem; font-weight: 700;">Restock Delivery</span>
            </div>
            <div style="font-size: 0.85rem; color: #b5c7b8;">
                Receive an incoming stock shipment from a pharmaceutical distributor for an existing formulary medicine.
            </div>
        </div>
        """)

        prod_batch_map = {f"{p['name']} ({p['generic_name']})": p for p in products_db}
        sel_b_label = st.selectbox("Select Existing Medicine to Restock", options=list(prod_batch_map.keys()), key="rec_med_sel")
        selected_b_prod = prod_batch_map[sel_b_label]

        # Live stock peek for this medicine from SQLite
        conn_live = get_connection()
        curr_lots = conn_live.execute("""
            SELECT batch_id, quantity_on_hand, expiry_date 
            FROM inventory_batches 
            WHERE product_id = ? AND quantity_on_hand > 0 
            ORDER BY expiry_date ASC;
        """, (selected_b_prod['product_id'],)).fetchall()
        conn_live.close()

        total_curr_stock = sum(l['quantity_on_hand'] for l in curr_lots)
        lots_str = ", ".join(f"<code>{l['batch_id']}</code> ({l['quantity_on_hand']:.0f}u, exp {l['expiry_date']})" for l in curr_lots) if curr_lots else "No active batches currently"
        render_html(f"""
        <div style="background: rgba(0,0,0,0.22); padding: 12px 16px; border-radius: 10px; margin: 10px 0 16px 0; border: 1px solid rgba(237, 230, 216, 0.1);">
            <div style="font-size: 0.75rem; color: #a4b7a7; text-transform: uppercase; font-weight: 600;">Current Stock on Shelf</div>
            <div style="font-size: 0.92rem; color: #f7f2e8; margin-top: 3px;">
                Total: <strong>{total_curr_stock:,.0f} units</strong> across {len(curr_lots)} batch(es): {lots_str}
            </div>
        </div>
        """)

        with st.form("receive_batch_existing_form"):
            rc1, rc2, rc3 = st.columns(3)
            rec_lot = rc1.text_input("Incoming Batch LOT ID", placeholder="e.g., LOT-DOLO-2020-B2", key="rec_lot_input")
            rec_qty = rc2.number_input("Units Received", min_value=1.0, value=100.0, step=10.0, key="rec_qty_input")
            rec_dt = rc3.date_input("Received Date", value=as_of_date, key="rec_dt_input")

            rec_exp = st.date_input("Manufacturer Expiry Date", value=date(2021, 12, 31), key="rec_exp_input")

            submit_batch = st.form_submit_button("📥 Receive Batch & Add to Physical Stock", use_container_width=True)
            if submit_batch:
                if not rec_lot.strip():
                    st.error("Please enter a valid Batch LOT ID.")
                elif rec_exp <= rec_dt:
                    st.error("Expiry date must be in the future relative to delivery date!")
                else:
                    try:
                        add_batch(
                            batch_id=rec_lot.strip(),
                            product_id=selected_b_prod['product_id'],
                            initial_qty=rec_qty,
                            received_date=str(rec_dt),
                            expiry_date=str(rec_exp)
                        )
                        st.success(f"✅ Successfully received Batch {rec_lot.strip()} ({rec_qty:.0f} units) of {selected_b_prod['name']} into dispensary inventory!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error receiving batch: {e}")

    # -------------------------------------------------------------------------
    # VENUE 3: REGISTER BRAND NEW MEDICINE & CATEGORY
    # -------------------------------------------------------------------------
    elif st.session_state.pos_active_drawer == "medicine":
        render_html("""
        <div class="pos-glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="margin: 0; font-size: 1.35rem; color: #fbf7ef;">➕ Operation 3: Register Brand New Medicine</h3>
                <span class="electric-pill">Formulary Expansion</span>
            </div>
            <div style="font-size: 0.85rem; color: #b5c7b8;">
                Registering a medicine brand not yet in the dispensary catalog? Enter its formulation details, choose its clinical category, and receive its initial delivery batch.
            </div>
        </div>
        """)

        with st.form("register_new_medicine_form"):
            new_name = st.text_input("Medicine Brand Name", placeholder="e.g., Augmentin 625 Duo Tablet", key="new_med_name")
            new_generic = st.text_input("Generic Active Salt & Strength", placeholder="e.g., Amoxycillin 500mg + Clavulanic Acid 125mg", key="new_med_salt")

            category_options = {
                "Analgesics (Paracetamol & Anilides) [Fever & Mild Pain]": ("N02BE", "N02BE_Analgesics_Paracetamol_Anilides"),
                "Anti-Inflammatory (Propionic) [Ibuprofen / Body Ache]": ("M01AE", "M01AE_Anti_Inflammatory_Propionic"),
                "Anti-Inflammatory (Acetic) [Diclofenac / Joint Pain]": ("M01AB", "M01AB_Anti_Inflammatory_Acetic"),
                "Analgesics (Salicylic) [Aspirin / Cardio]": ("N02BA", "N02BA_Analgesics_Salicylic"),
                "Antihistamines Systemic [Allergy / Cold / Hives]": ("R06", "R06_Antihistamines_Systemic"),
                "Respiratory & Airway [Asthma / Inhalers / Cough]": ("R03", "R03_Respiratory_Airway_Diseases"),
                "Psycholeptics (Anxiolytics) [Anti-Anxiety / Calming]": ("N05B", "N05B_Psycholeptics_Anxiolytics"),
                "Psycholeptics (Hypnotics & Sedatives) [Sleep Aids]": ("N05C", "N05C_Psycholeptics_Hypnotics_Sedatives")
            }
            selected_cat_label = st.selectbox("Select Therapeutic Category", options=list(category_options.keys()), key="new_med_cat")

            nb_c1, nb_c2, nb_c3 = st.columns(3)
            new_lot = nb_c1.text_input("Initial Batch LOT ID", value="LOT-IND-2020-01", key="new_med_lot")
            new_qty = nb_c2.number_input("Units Received", min_value=1.0, value=100.0, step=10.0, key="new_med_qty")
            new_exp = nb_c3.date_input("Batch Expiry Date", value=date(2021, 12, 31), key="new_med_exp")

            submit_new_med = st.form_submit_button("💾 Save Medicine & Receive Initial Batch", use_container_width=True)
            if submit_new_med:
                if not new_name.strip() or not new_generic.strip() or not new_lot.strip():
                    st.error("Please provide Medicine Brand Name, Generic Salt, and initial Batch LOT.")
                elif new_exp <= as_of_date:
                    st.error("Expiry date must be in the future relative to the cutoff date!")
                else:
                    try:
                        atc, cat_full = category_options[selected_cat_label]
                        new_p_id = add_product(
                            name=new_name.strip(),
                            generic_name=new_generic.strip(),
                            atc_code=atc,
                            category_full_name=cat_full,
                            brand_share=0.10
                        )
                        add_batch(
                            batch_id=new_lot.strip(),
                            product_id=new_p_id,
                            initial_qty=new_qty,
                            received_date=str(as_of_date),
                            expiry_date=str(new_exp)
                        )
                        st.success(f"✅ Successfully registered {new_name.strip()} under {selected_cat_label.split(' [')[0]} with batch {new_lot.strip()}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error registering medicine: {e}")
