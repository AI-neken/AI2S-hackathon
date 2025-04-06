#!/usr/bin/env python

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from statsforecast import StatsForecast
from statsforecast.models import HoltWinters, AutoARIMA, AutoETS, MSTL, IMAPA
from utils import preprocess

# Configuration
INPUT_PATH = Path('data/01_input_history.csv')
OUTPUT_PATH = Path('outputs/01_output_prediction_1239.csv')
FORECAST_HORIZON = 12
SEASON_LENGTH = 12

def restore_original_format(date_column: pd.Series) -> pd.Series:
    """Convert datetime series to string format 'MonthYear'."""
    return date_column.dt.strftime('%b%Y')

def create_zero_forecast_data(null_ids: List[str], inactive_ids: List[str], start_date: str) -> pd.DataFrame:
    """Generate zero-quantity forecast for null and inactive IDs."""
    zero_ids = null_ids + inactive_ids
    date_range = pd.date_range(start=start_date, periods=FORECAST_HORIZON, freq='MS')
    
    return pd.DataFrame({
        'unique_id': np.repeat(zero_ids, len(date_range)),
        'ds': np.tile(date_range, len(zero_ids)),
        'Quantity': 0
    })

def add_country_product_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add Country and Product columns extracted from unique_id."""
    split_df = df['unique_id'].str.split('_', expand=True)
    df['Country'] = split_df[0]
    df['Product'] = split_df[1]
    return df

def setup_forecast_model() -> StatsForecast:
    """Initialize ensemble forecasting model."""
    return StatsForecast(
        models=[
            HoltWinters(season_length=SEASON_LENGTH, error_type="A"),
            MSTL(season_length=SEASON_LENGTH),
            IMAPA(),
            AutoETS(season_length=SEASON_LENGTH),
            AutoARIMA(season_length=SEASON_LENGTH),
        ],
        freq='MS',
        n_jobs=-1
    )

def main() -> None:
    """Main function for ensemble forecasting and result export."""
    # Load and preprocess data
    df = pd.read_csv(INPUT_PATH)
    df_null, df_inactive, df_active = preprocess.preprocess_ex1_final_sub(df)

    # Setup and fit forecast model
    sf = setup_forecast_model()
    sf.fit(
        df_active[['unique_id', 'ds', 'Quantity']],
        target_col='Quantity'
    )

    # Predict and calculate ensemble average
    forecasts = sf.predict(h=FORECAST_HORIZON)
    forecasts['EnsembleAverage'] = forecasts[['HoltWinters', 'MSTL', 'IMAPA', 'AutoETS', 'AutoARIMA']].mean(axis=1).astype(int)
    
    forecasts = forecasts[['unique_id', 'ds', 'EnsembleAverage']].rename(columns={'EnsembleAverage': 'Quantity'})

    # Create zero forecasts
    df_zeros = create_zero_forecast_data(
        null_ids=list(df_null['unique_id'].unique()),
        inactive_ids=list(df_inactive['unique_id'].unique()),
        start_date='2024-01-01'
    )

    # Combine all forecasts
    final_df = pd.concat([forecasts, df_zeros], ignore_index=True)

    # Add Month, Country, Product columns
    final_df['Month'] = restore_original_format(final_df['ds'])
    final_df = add_country_product_columns(final_df)

    # Reorder and export
    final_df = final_df[['Country', 'Product', 'Month', 'Quantity']]
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Forecast saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
