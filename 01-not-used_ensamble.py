#!/usr/bin/env python

import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import HoltWinters, AutoARIMA, AutoETS, MSTL, IMAPA
from utils import preprocess
from tqdm import tqdm
from typing import List
from pathlib import Path
from datetime import datetime

# Configuration
INPUT_PATH = Path('data/01_input_history.csv')
VALIDATION_OUTPUT_PATH = Path('outputs/validation_classical.csv')
SUBMISSION_AVG_PATH = Path('outputs/submission_Average.csv')
SUBMISSION_MEDIAN_PATH = Path('outputs/submission_Median.csv')
FORECAST_HORIZON = 12

def restore_original_format(date_column: pd.Series) -> pd.Series:
    """Convert datetime series to string format 'MonthYear'."""
    return date_column.dt.strftime('%b%Y')

def add_ensemble_predictions(
    df_validation: pd.DataFrame,
    forecasts: pd.DataFrame,
    active_ids: List[str]
) -> pd.DataFrame:
    """Add ensemble average and median predictions to the validation set."""
    df_validation['Average'] = 0
    df_validation['Median'] = 0

    for uid in tqdm(active_ids):
        df_validation.loc[df_validation['unique_id'] == uid, 'Average'] = forecasts.loc[
            forecasts['unique_id'] == uid, 'EnsambleAverage'
        ].values.astype(int)

        df_validation.loc[df_validation['unique_id'] == uid, 'Median'] = forecasts.loc[
            forecasts['unique_id'] == uid, 'EnsambleMedian'
        ].values.astype(int)

    return df_validation

def save_forecasts(df_validation: pd.DataFrame) -> None:
    """Save validation and submission forecasts to CSV files."""
    df_validation['Month'] = restore_original_format(df_validation['ds'])
    
    # Save validation with ground truth
    df_validation[['Quantity', 'Product', 'Month', 'Country']].to_csv(VALIDATION_OUTPUT_PATH, index=False)

    # Save average ensemble submission
    df_forecast_avg = df_validation[['Average', 'Country', 'Product', 'Month']].copy()
    df_forecast_avg.rename(columns={'Average': 'Quantity'}, inplace=True)
    df_forecast_avg[['Quantity', 'Product', 'Month', 'Country']].to_csv(SUBMISSION_AVG_PATH, index=False)

    # Save median ensemble submission
    df_forecast_med = df_validation[['Median', 'Country', 'Product', 'Month']].copy()
    df_forecast_med.rename(columns={'Median': 'Quantity'}, inplace=True)
    df_forecast_med[['Quantity', 'Product', 'Month', 'Country']].to_csv(SUBMISSION_MEDIAN_PATH, index=False)

    print(f"Forecasts saved to:\n- {VALIDATION_OUTPUT_PATH}\n- {SUBMISSION_AVG_PATH}\n- {SUBMISSION_MEDIAN_PATH}")

def main() -> None:
    """Main execution function."""
    # Load and preprocess data
    df = pd.read_csv(INPUT_PATH)
    df_train_null, df_train_inactive, df_train_active, df_validation = preprocess.preprocess_ex1(df)

    # Initialize and fit model
    sf = StatsForecast(
        models=[
            HoltWinters(season_length=12, error_type="A"),
            MSTL(season_length=12),
            IMAPA(),
            AutoETS(season_length=12),
            AutoARIMA(season_length=12),
        ],
        freq='MS',
        n_jobs=-1
    )
    sf.fit(df_train_active[['unique_id', 'ds', 'Quantity']], target_col='Quantity')

    # Generate forecasts and compute ensembles
    forecasts = sf.predict(h=FORECAST_HORIZON)
    model_cols = ['HoltWinters', 'MSTL', 'IMAPA', 'AutoETS', 'AutoARIMA']
    forecasts['EnsambleAverage'] = forecasts[model_cols].mean(axis=1)
    forecasts['EnsambleMedian'] = forecasts[model_cols].median(axis=1)

    # Add ensemble results to validation set
    df_validation = add_ensemble_predictions(df_validation, forecasts, list(df_train_active['unique_id'].unique()))

    # Save results
    save_forecasts(df_validation)

if __name__ == "__main__":
    main()
