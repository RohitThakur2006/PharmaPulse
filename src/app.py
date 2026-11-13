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

from engine import evaluate_reorder_risk, evaluate_batch_expiry_fefo
from features import prepare_inference_row, FEATURE_COLUMNS, CATEGORY_MAPPING

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

    demand_lower = max(0.0, medicine_demand_28d * 0.90)
    demand_upper = medicine_demand_28d * 1.10

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
        safety_stock_days=5
    )

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

    cat_friendly = p_cat.split('_', 1)[1].replace('_', ' ') if '_' in p_cat else p_cat
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
                        <div class="triage-subtitle">Generic: <strong>{item['generic']}</strong> &nbsp;•&nbsp; ATC: {item['atc_code']} ({item['category'].split('_', 1)[1].replace('_', ' ') if '_' in item['category'] else item['category']})</div>
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
                        <div style="font-size: 0.72rem; color: #8fa593; margin-top: 2px;">Expected range: {item['demand_lower']:.0f} – {item['demand_upper']:.0f}</div>
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

        # Render Custom Glass Table
        table_rows_html = ""
        for r in displayed_catalog:
            if r['has_expired']:
                status_pill = f"<span class='badge-critical' style='background-color: rgba(220, 38, 38, 0.3); border-color: #ef4444; color: #fca5a5;'>⚫ {r['expired_units']:.0f} Expired</span>"
            elif r['has_future_waste']:
                status_pill = f"<span class='badge-critical'>🔴 ~{r['future_waste']:.0f} Units Risk</span>"
            elif r['runway'] < 7.0:
                status_pill = "<span class='badge-warning'>🟡 Stockout Warning</span>"
            else:
                status_pill = "<span class='badge-optimal'>🟢 Balanced</span>"

            runway_color = "#ef4444" if r['runway'] < 7.0 else "#f59e0b" if r['runway'] < 12.0 else "#10b981"

            table_rows_html += f"""
            <tr>
                <td>
                    <strong style="color: #fbf7ef; font-size: 0.95rem;">{r['name']}</strong>
                    <div style="font-size: 0.78rem; color: #a4b7a7; margin-top: 2px;">{r['generic']}</div>
                </td>
                <td><span style="background: rgba(255, 255, 255, 0.06); padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; border: 1px solid rgba(237, 230, 216, 0.12); color: #c5d4c8;">{r['category']}</span></td>
                <td><strong>{r['usable_stock']:,.0f}</strong> <span style="font-size: 0.78rem; color: #a4b7a7;">units</span></td>
                <td>~{r['demand_28d']:.0f} <span style="font-size: 0.78rem; color: #a4b7a7;">units</span></td>
                <td><span style="color: {runway_color}; font-weight: 700;">{r['runway']:.1f} days</span></td>
                <td>{status_pill}</td>
            </tr>
            """

        custom_inventory_table_html = f"""
        <table class="custom-table">
            <thead>
                <tr>
                    <th>Medicine & Formulation</th>
                    <th>Therapeutic Category</th>
                    <th>Usable Stock</th>
                    <th>28d Forecast</th>
                    <th>Stock Runway</th>
                    <th>Dispensary Status</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html if table_rows_html else "<tr><td colspan='6' style='text-align: center; color: #a4b7a7;'>No medicines match your search criteria.</td></tr>"}
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
                <td><code style="background: rgba(0, 0, 0, 0.3); padding: 3px 8px; border-radius: 6px; color: #ede4d3; font-family: 'JetBrains Mono', monospace;">{b['batch_id']}</code></td>
                <td><strong>{b['product_name']}</strong><div style="font-size: 0.76rem; color: #a4b7a7;">{b['generic_name']}</div></td>
                <td><strong>{b['quantity_on_hand']:,.0f}</strong> / {b['initial_quantity']:,.0f} units</td>
                <td>{b['received_date']}</td>
                <td>{b['expiry_date']}</td>
                <td>{days_str}</td>
                <td>{exp_badge}</td>
            </tr>
            """

        batches_table_html = f"""
        <table class="custom-table">
            <thead>
                <tr>
                    <th>Batch LOT</th>
                    <th>Medicine Formulation</th>
                    <th>Units Remaining</th>
                    <th>Received Date</th>
                    <th>Expiry Date</th>
                    <th>Shelf Life Left</th>
                    <th>FEFO Status</th>
                </tr>
            </thead>
            <tbody>
                {batch_rows_html if batch_rows_html else "<tr><td colspan='7' style='text-align: center; color: #a4b7a7;'>No batches match your filter criteria.</td></tr>"}
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

    # DRAWER 1: DAILY DISPENSE COUNTER
    with st.expander("📝 Operation 1: Daily Dispense Counter (Patient Sale)", expanded=(st.session_state.pos_active_drawer == "dispense")):
        render_html("""
        <div style="margin-bottom: 14px;">
            <div style="font-size: 0.88rem; color: #b5c7b8;">
                Record patient dispensing. Physical stock is atomically deducted from the oldest valid batch in <strong>First-Expiring, First-Out (FEFO)</strong> sequence.
            </div>
        </div>
        """)
        with st.form("pos_sale_form"):
            sale_date_val = st.date_input("Sale Date", value=as_of_date, key="sale_dt_pos")
            
            # Select exact medicine
            prod_map = {f"{p['name']} ({p['generic_name']})": p for p in products_db}
            sel_label = st.selectbox("Select Exact Medicine Dispensed", options=list(prod_map.keys()), key="sale_med_pos")
            selected_prod = prod_map[sel_label]
            
            # Check available unexpired batches for this product
            active_p_batches = batches_df[
                (batches_df['product_id'] == selected_prod['product_id']) & 
                (batches_df['expiry_date'] > str(as_of_date))
            ].sort_values('expiry_date')
            total_avail = active_p_batches['quantity_on_hand'].sum() if not active_p_batches.empty else 0

            # Unit input
            dispensed_units = st.number_input(
                f"Units to Dispense (Usable Stock on shelf: {total_avail:,.0f} units)", 
                min_value=1.0, 
                value=min(15.0, max(1.0, float(total_avail))), 
                step=1.0,
                key="sale_units_pos"
            )

            # Peek oldest batch
            if not active_p_batches.empty:
                oldest_b = active_p_batches.iloc[0]
                days_left = (pd.to_datetime(oldest_b['expiry_date']) - pd.to_datetime(as_of_date)).days
                render_html(f"""
                <div style="background: rgba(0,0,0,0.25); padding: 12px 16px; border-radius: 12px; margin: 14px 0; border: 1px solid rgba(237, 230, 216, 0.12);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.74rem; color: #a4b7a7; text-transform: uppercase; font-weight: 700; letter-spacing: 0.6px;">FEFO Queue Front Batch</span>
                        <span class="electric-pill">Next in Line</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #f7f2e8; margin-top: 6px;">
                        Batch LOT: <code>{oldest_b['batch_id']}</code> &nbsp;•&nbsp; Stock: <strong>{oldest_b['quantity_on_hand']:.0f} units</strong> &nbsp;•&nbsp; Expires: {oldest_b['expiry_date']} ({days_left}d left)
                    </div>
                </div>
                """)
            else:
                render_html("""
                <div style="background: rgba(220, 38, 38, 0.15); padding: 12px 16px; border-radius: 12px; margin: 14px 0; border: 1px solid rgba(239, 68, 68, 0.35);">
                    <span style="color: #fca5a5; font-size: 0.88rem; font-weight: 600;">⚠️ ZERO usable batches in stock! Dispensing this medicine will record unmet demand (stockout).</span>
                </div>
                """)

            submit_dispense = st.form_submit_button("⚡ Dispense & Deduct from FEFO Stock", use_container_width=True)
            if submit_dispense:
                try:
                    res = log_sale_atomic(
                        product_id=selected_prod['product_id'],
                        sale_date=str(sale_date_val),
                        units_requested=dispensed_units
                    )
                    if res['unmet_demand'] > 0:
                        st.warning(f"⚠️ Dispensed {res['dispensed']:.0f} units of {selected_prod['name']}. Unmet demand of {res['unmet_demand']:.0f} units recorded due to stockout!")
                    else:
                        st.success(f"✅ Successfully dispensed {res['dispensed']:.0f} units of {selected_prod['name']} via FEFO depletion!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error logging sale: {e}")

    # DRAWER 2: RECEIVE NEW BATCH FOR EXISTING MEDICINE
    with st.expander("📦 Operation 2: Receive New Batch (Existing Medicine Stock Replenishment)", expanded=(st.session_state.pos_active_drawer == "batch")):
        render_html("""
        <div style="margin-bottom: 14px;">
            <div style="font-size: 0.88rem; color: #b5c7b8;">
                Receive incoming stock shipment for a medicine already registered in the formulary. The new batch is immediately queued into physical inventory.
            </div>
        </div>
        """)
        with st.form("receive_batch_existing_form"):
            prod_batch_map = {f"{p['name']} ({p['generic_name']})": p for p in products_db}
            sel_b_label = st.selectbox("Select Existing Medicine in Dispensary", options=list(prod_batch_map.keys()), key="rec_med_sel")
            selected_b_prod = prod_batch_map[sel_b_label]

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
                    st.error("Expiry date must be in the future relative to the delivery date!")
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

    # DRAWER 3: REGISTER BRAND NEW MEDICINE & CATEGORY
    with st.expander("➕ Operation 3: Register Brand New Medicine (Formulary Expansion)", expanded=(st.session_state.pos_active_drawer == "medicine")):
        render_html("""
        <div style="margin-bottom: 14px;">
            <div style="font-size: 0.88rem; color: #b5c7b8;">
                Adding a new medicine brand not yet in the dispensary catalog? Enter its name, generic active salt, choose its therapeutic category, and register its initial batch.
            </div>
        </div>
        """)
        with st.form("register_new_medicine_form"):
            new_name = st.text_input("Medicine Brand Name", placeholder="e.g., Augmentin 625 Duo Tablet", key="new_med_name")
            new_generic = st.text_input("Generic Active Salt & Strength", placeholder="e.g., Amoxycillin 500mg + Clavulanic Acid 125mg", key="new_med_salt")

            # Explicit category choices for the user
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
