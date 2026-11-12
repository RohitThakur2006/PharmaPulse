"""
engine.py — Core Decision Support Logic.
Pure computation engine with zero SQL or Streamlit dependencies.
Implements:
1. Cumulative FEFO batch allocation
2. Independent Reorder & Expiry alert flags
3. Uncertainty bands (+/- MAE)
4. Configurable supplier lead times
"""

from typing import List, Dict, Any
import numpy as np

# Uncertainty margin from our validated CatBoost model (test MAE = 35.13 units)
MODEL_MAE = 35.13

def evaluate_reorder_risk(
    product_name: str,
    total_stock_on_hand: float,
    predicted_28d_demand: float,
    lead_time_days: int = 7,
    safety_stock_days: int = 5
) -> Dict[str, Any]:
    """
    Evaluates stockout / reorder need based on demand forecast and supplier lead time.
    
    Formula:
    Daily Run-rate = predicted_28d_demand / 28
    Days of Inventory (DOI) = total_stock_on_hand / Daily Run-rate
    Critical if stock covers less than the delivery lead time; Reorder if stock
    cannot cover the full 28-day forecast.
    """
    daily_rate = max(0.01, predicted_28d_demand / 28.0)
    doi = total_stock_on_hand / daily_rate

    shortage_28d = max(0.0, predicted_28d_demand - total_stock_on_hand)

    # Uncertainty bounds
    demand_low = max(0.0, predicted_28d_demand - MODEL_MAE)
    demand_high = predicted_28d_demand + MODEL_MAE

    if total_stock_on_hand < (lead_time_days * daily_rate):
        flag = "CRITICAL_STOCKOUT"
        badge = "🔴 Critical Reorder Needed"
        guidance = (
            f"Stock will run out in ~{doi:.1f} days, before standard {lead_time_days}-day delivery! "
            f"Recommended order: ~{shortage_28d + (safety_stock_days * daily_rate):.0f} units immediately."
        )
    elif total_stock_on_hand < predicted_28d_demand:
        flag = "REORDER_SOON"
        badge = "🟡 Reorder Warning"
        guidance = (
            f"Current stock covers ~{doi:.1f} days, less than the 28-day forecast. "
            f"Plan replenishment of ~{shortage_28d + (safety_stock_days * daily_rate):.0f} units within the week."
        )
    elif total_stock_on_hand > (predicted_28d_demand * 2.5):
        flag = "OVERSTOCKED"
        badge = "🔵 Overstocked (Safe)"
        guidance = (
            f"Stock covers ~{doi:.1f} days (~{doi/30:.1f} months). "
            f"Capital is locked; avoid placing fresh purchase orders."
        )
    else:
        flag = "OPTIMAL"
        badge = "🟢 Optimal Stock"
        guidance = f"Stock covers ~{doi:.1f} days. Sufficient coverage for expected demand."

    return {
        'product_name': product_name,
        'total_stock': total_stock_on_hand,
        'forecast_28d': round(predicted_28d_demand, 1),
        'forecast_range': (round(demand_low, 1), round(demand_high, 1)),
        'days_of_inventory': round(doi, 1),
        'reorder_flag': flag,
        'reorder_badge': badge,
        'reorder_guidance': guidance,
        'suggested_reorder_qty': round(max(0.0, shortage_28d), 0)
    }

def evaluate_batch_expiry_fefo(
    batches: List[Dict[str, Any]],
    predicted_28d_demand: float,
    as_of_date_str: str,
    horizon_days: int = 28
) -> List[Dict[str, Any]]:
    """
    Evaluates individual batch expiry risk using CUMULATIVE FEFO allocation.
    
    Earlier expiring batches consume customer demand first.
    Later batches only get whatever demand remains after earlier batches are filled.
    
    Each batch in `batches` must contain:
    - 'batch_id': str
    - 'quantity_on_hand': float
    - 'expiry_date': str (YYYY-MM-DD)
    """
    import pandas as pd
    as_of = pd.to_datetime(as_of_date_str)
    daily_rate = max(0.01, predicted_28d_demand / 28.0)

    # Sort batches strictly in FEFO order (earliest expiry first)
    sorted_batches = sorted(batches, key=lambda b: pd.to_datetime(b['expiry_date']))

    cumulative_demand_used = 0.0
    evaluated = []

    for b in sorted_batches:
        exp_date = pd.to_datetime(b['expiry_date'])
        days_to_expiry = max(0, (exp_date - as_of).days)
        stock = b['quantity_on_hand']

        # Stock already past its expiry date is waste, not a forecast
        if exp_date < as_of:
            evaluated.append({
                'batch_id': b['batch_id'],
                'expiry_date': b['expiry_date'],
                'days_to_expiry': days_to_expiry,
                'current_stock': stock,
                'forecast_demand_window': 0.0,
                'expected_waste_units': stock,
                'expiry_flag': 'EXPIRED',
                'expiry_badge': '⚫ Already Expired',
                'advisory_action': f'Remove from shelf: batch expired on {exp_date.date()}.'
            })
            continue

        # Batches expiring beyond the horizon are safe for now
        if days_to_expiry > horizon_days:
            evaluated.append({
                'batch_id': b['batch_id'],
                'expiry_date': b['expiry_date'],
                'days_to_expiry': days_to_expiry,
                'current_stock': stock,
                'forecast_demand_window': round(daily_rate * days_to_expiry, 1),
                'expected_waste_units': 0.0,
                'expiry_flag': 'SAFE',
                'expiry_badge': '🟢 Safe Expiry',
                'advisory_action': 'Expiry date is beyond immediate planning horizon.'
            })
            continue

        # Total demand expected between today and this batch's expiry date
        demand_until_expiry = daily_rate * days_to_expiry

        # Demand left available for THIS batch after earlier batches were satisfied
        demand_available = max(0.0, demand_until_expiry - cumulative_demand_used)

        # Expected waste = stock that cannot be absorbed by available demand
        expected_waste = max(0.0, stock - demand_available)
        demand_absorbed = min(stock, demand_available)
        cumulative_demand_used += demand_absorbed

        if expected_waste > 0:
            if days_to_expiry <= 10:
                flag = 'CRITICAL_WASTE'
                badge = '🔴 Critical Expiry Risk'
                action = (
                    f"Operational Guidance: ~{expected_waste:.0f} units projected to expire unsold in {days_to_expiry} days. "
                    f"Consider immediate clearance discount, front-shelf rotation, or B2B distributor return."
                )
            else:
                flag = 'MODERATE_WASTE'
                badge = '🟡 Moderate Expiry Warning'
                action = (
                    f"Operational Guidance: ~{expected_waste:.0f} units at risk of expiring in {days_to_expiry} days. "
                    f"Monitor dispensing rate and prioritize dispensing this batch."
                )
        else:
            flag = 'SAFE'
            badge = '🟢 Safe (Will Clear)'
            action = f"Expected to clear safely via FEFO walk-in sales prior to {exp_date.date()}."

        evaluated.append({
            'batch_id': b['batch_id'],
            'expiry_date': b['expiry_date'],
            'days_to_expiry': days_to_expiry,
            'current_stock': stock,
            'forecast_demand_window': round(demand_until_expiry, 1),
            'expected_waste_units': round(expected_waste, 1),
            'expiry_flag': flag,
            'expiry_badge': badge,
            'advisory_action': action
        })

    return evaluated
