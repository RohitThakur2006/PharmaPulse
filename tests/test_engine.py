"""
test_engine.py — Unit tests verifying engine.py against the 5 real-world scenarios.
"""

import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from engine import evaluate_reorder_risk, evaluate_batch_expiry_fefo

def test_scenario_1_optimal_stock():
    # Demand = 300, Stock = 320 -> Optimal, no reorder needed
    res = evaluate_reorder_risk(
        product_name="Paracetamol",
        total_stock_on_hand=320,
        predicted_28d_demand=300,
        lead_time_days=7,
        safety_stock_days=5
    )
    assert res['reorder_flag'] == 'OPTIMAL'
    assert res['suggested_reorder_qty'] == 0.0

def test_scenario_2_reorder_needed():
    # Demand = 320, Stock = 200 -> Shortage of ~120 units
    res = evaluate_reorder_risk(
        product_name="Paracetamol",
        total_stock_on_hand=200,
        predicted_28d_demand=320,
        lead_time_days=7,
        safety_stock_days=5
    )
    assert res['reorder_flag'] in ['REORDER_SOON', 'CRITICAL_STOCKOUT']
    assert res['suggested_reorder_qty'] == 120.0

def test_scenario_3_cumulative_fefo_expiry():
    # Demand = 280 over 28 days = 10 units/day
    # Batch 1: 50 units, expires in 10 days (demand in 10 days = 100 units -> Batch 1 clears safely)
    # Batch 2: 120 units, expires in 15 days (demand in 15 days = 150 units. Batch 1 took 50 -> 100 left for Batch 2)
    # -> Batch 2 has 120 units but only 100 demand left -> 20 units WASTE!
    batches = [
        {'batch_id': 'B1', 'quantity_on_hand': 50, 'expiry_date': '2026-12-11'},
        {'batch_id': 'B2', 'quantity_on_hand': 120, 'expiry_date': '2026-12-16'}
    ]
    res = evaluate_batch_expiry_fefo(
        batches=batches,
        predicted_28d_demand=280.0,
        as_of_date_str='2026-12-01',
        horizon_days=28
    )
    assert len(res) == 2
    assert res[0]['batch_id'] == 'B1'
    assert res[0]['expiry_flag'] == 'SAFE'
    assert res[0]['expected_waste_units'] == 0.0

    assert res[1]['batch_id'] == 'B2'
    assert res[1]['expected_waste_units'] == 20.0
    assert 'WASTE' in res[1]['expiry_flag']
