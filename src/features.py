"""
features.py — Shared feature engineering for training and inference.
Guarantees zero train/serve skew by using the exact same logic.
"""

import pandas as pd
import numpy as np

# Official mapping of ATC codes to human-readable category names
CATEGORY_MAPPING = {
    'M01AB': 'M01AB_Anti_Inflammatory_Acetic',
    'M01AE': 'M01AE_Anti_Inflammatory_Propionic',
    'N02BA': 'N02BA_Analgesics_Salicylic',
    'N02BE': 'N02BE_Analgesics_Paracetamol_Anilides',
    'N05B':  'N05B_Psycholeptics_Anxiolytics',
    'N05C':  'N05C_Psycholeptics_Hypnotics_Sedatives',
    'R03':   'R03_Respiratory_Airway_Diseases',
    'R06':   'R06_Antihistamines_Systemic'
}

# The 8 reference category names
ALL_CATEGORIES = list(CATEGORY_MAPPING.values())

# Exact feature order the saved CatBoost artifact was trained on (see model_metadata.json)
FEATURE_COLUMNS = [
    'sales_last_1w', 'sales_last_2w', 'sales_last_4w', 'sales_same_week_last_year',
    'sales_last_28d', 'avg_last_90d', 'avg_last_365d', 'volatility_weekly_sales',
    'season_Spring', 'season_Summer', 'season_Winter',
    'category_M01AE_Anti_Inflammatory_Propionic',
    'category_N02BA_Analgesics_Salicylic',
    'category_N02BE_Analgesics_Paracetamol_Anilides',
    'category_N05B_Psycholeptics_Anxiolytics',
    'category_N05C_Psycholeptics_Hypnotics_Sedatives',
    'category_R03_Respiratory_Airway_Diseases',
    'category_R06_Antihistamines_Systemic',
]

def get_season(month: int) -> str:
    """Map calendar month to meteorological season."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def build_features_for_category(category_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all lag, rolling, and seasonal features for a single category's
    weekly sales history sorted chronologically.
    
    Expects columns: ['week_start', 'weekly_sales']
    """
    g = category_weekly.copy().sort_values('week_start').reset_index(drop=True)

    # Past lags (shift >= 1 guarantees no future leakage)
    g['sales_last_1w'] = g['weekly_sales'].shift(1)
    g['sales_last_2w'] = g['weekly_sales'].shift(2)
    g['sales_last_4w'] = g['weekly_sales'].shift(4)
    g['sales_same_week_last_year'] = g['weekly_sales'].shift(52)

    # Rolling statistics from completed weeks
    past = g['weekly_sales'].shift(1)
    g['sales_last_28d'] = past.rolling(4).sum()
    g['avg_last_90d']   = past.rolling(12).mean()
    g['avg_last_365d']  = past.rolling(52).mean()
    g['volatility_weekly_sales'] = past.rolling(12).std()

    # Calendar & Season
    g['month'] = g['week_start'].dt.month
    g['season'] = g['month'].apply(get_season)

    return g

def prepare_inference_row(history_df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    Given a daily sales history table up to `as_of_date`,
    prepares the exact 18 feature columns required by CatBoost for all 8 categories.
    
    Expected input columns: ['datum', 'category', 'units_sold']
    """
    # 1. Filter strictly up to as_of_date (yesterday cutoff)
    df = history_df.copy()
    df['datum'] = pd.to_datetime(df['datum'])
    df = df[df['datum'] <= pd.to_datetime(as_of_date)]

    rows = []
    for cat in ALL_CATEGORIES:
        cat_df = df[df['category'] == cat]
        
        # Resample daily to weekly (Monday-start)
        cat_weekly = (
            cat_df.groupby(pd.Grouper(key='datum', freq='W-MON'))['units_sold']
            .sum()
            .reset_index()
            .rename(columns={'datum': 'week_start', 'units_sold': 'weekly_sales'})
        )
        
        # Build features
        featured = build_features_for_category(cat_weekly)
        
        # Grab the latest complete row (as-of prediction point)
        latest_row = featured.iloc[[-1]].copy()
        latest_row['category'] = cat
        rows.append(latest_row)

    combined = pd.concat(rows, axis=0).reset_index(drop=True)

    # One-hot encode seasons and categories to match training schema
    for season in ['Fall', 'Spring', 'Summer', 'Winter']:
        combined[f'season_{season}'] = (combined['season'] == season).astype(int)

    for cat in ALL_CATEGORIES:
        # Note: M01AB was the dropped first category in training
        if cat != 'M01AB_Anti_Inflammatory_Acetic':
            combined[f'category_{cat}'] = (combined['category'] == cat).astype(int)

    return combined
