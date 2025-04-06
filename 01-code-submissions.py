#!/usr/bin/env python

import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import HoltWinters
from utils import preprocess
from typing import List, Tuple, Set
from pathlib import Path
from datetime import datetime

# Configuration
INPUT_PATH = Path('data/01_input_history.csv')
OUTPUT_PATH = Path('outputs/01_output_prediction_1239.csv')
FORECAST_HORIZON = 12

def restore_original_format(date_column: pd.Series) -> pd.Series:
    """Convert datetime series to string format 'MonthYear'."""
    return date_column.dt.strftime('%b%Y')

def load_and_preprocess_data(input_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data and split into different categories based on activity patterns."""
    df = pd.read_csv(input_path)
    return preprocess.preprocess_ex1_final_sub(df)

def setup_forecast_model(season_length: int = 12) -> StatsForecast:
    """Initialize the forecasting model."""
    return StatsForecast(
        models=[
            HoltWinters(season_length=season_length, error_type="A"),
        ],
        freq='MS',
        n_jobs=-1
    )

def create_zero_forecast_data(
    null_ids: List[str], 
    inactive_ids: List[str], 
    forecast_periods: int, 
    start_date: datetime
) -> pd.DataFrame:
    """Create dataframe with zero forecasts for null and inactive IDs."""
    zero_ids: List[str] = null_ids + inactive_ids
    date_range = pd.date_range(start=start_date, periods=forecast_periods, freq='MS')
    
    return pd.DataFrame({
        'unique_id': np.repeat(zero_ids, len(date_range)),
        'ds': np.tile(date_range, len(zero_ids)),
        'Quantity': 0
    })

def add_country_product_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract country and product from unique_id and add as columns."""
    for id in df['unique_id'].unique():
        parts = id.split('_')
        df.loc[df['unique_id'] == id, 'Country'] = parts[0]
        df.loc[df['unique_id'] == id, 'Product'] = parts[1]
    return df

def main() -> None:
    """Main execution function."""
    
    # Load data
    df_train_null, df_train_inactive, df_train_active = load_and_preprocess_data(INPUT_PATH)
    
    # Set up and fit model
    sf = setup_forecast_model()
    sf.fit(
        df_train_active[['unique_id', 'ds', 'Quantity']],
        target_col='Quantity',
    )
    
    # Generate forecasts
    forecasts = sf.predict(h=FORECAST_HORIZON)
    forecasts['Quantity'] = forecasts[['HoltWinters']].astype(int)
    forecasts = forecasts.drop(columns='HoltWinters')
    
    # Handle null and inactive IDs
    zero_forecasts = create_zero_forecast_data(
        null_ids=list(df_train_null['unique_id'].unique()),
        inactive_ids=list(df_train_inactive['unique_id'].unique()),
        forecast_periods=FORECAST_HORIZON,
        start_date='2024-01-01'
    )
    
    # Combine forecasts
    final_df = pd.concat([forecasts, zero_forecasts], ignore_index=True)
    
    # Add month column in required format
    final_df['Month'] = restore_original_format(final_df['ds'])
    
    # Add country and product columns
    final_df = add_country_product_columns(final_df)
    
    # Save results
    final_df[['Country', 'Product', 'Month', 'Quantity']].to_csv(OUTPUT_PATH, index=False)
    
    print(f"Forecast successfully saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()