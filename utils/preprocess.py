import pandas as pd
import numpy as np
from typing import Tuple, List, Set, Dict, Any, Union, Optional
from datetime import datetime

def count_quantity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum quantities for each unique ID.
    
    Args:
        df: DataFrame with 'unique_id' and 'Quantity' columns
        
    Returns:
        DataFrame with unique IDs and their total quantities
    """
    return df.groupby('unique_id').agg({
        'Quantity': 'sum'
    }).reset_index()

def get_null_time_series(df: pd.DataFrame) -> np.ndarray:
    """
    Identify time series with total quantity of zero.
    
    Args:
        df: Input DataFrame with time series data
        
    Returns:
        Array of unique IDs with zero total quantity
    """
    df_grouped = count_quantity(df)
    return df_grouped[df_grouped['Quantity'] == 0]['unique_id'].values

def get_ending_zero_series(df: pd.DataFrame, final_submission: bool = True) -> np.ndarray:
    """
    Identify time series that end with zeros in the last year.
    
    Args:
        df: Input DataFrame with time series data
        final_submission: If True, use 2023 as last year, otherwise use 2022
        
    Returns:
        Array of unique IDs that end with zeros
    """
    null_list = get_null_time_series(df)
    last_year = 2023 if final_submission else 2022
    
    return df[
        (~df['unique_id'].isin(null_list)) & 
        (df['ds'].dt.year == last_year) & 
        (df['Quantity'] == 0)
    ]['unique_id'].values

def filter_by_active_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter time series data to include only periods between first and last non-zero entries.
    
    Args:
        df: Input DataFrame with time series data
        
    Returns:
        Filtered DataFrame with data only within active periods
    """
    # Only keep rows with non-zero quantity
    non_null_data = df[df['Quantity'] != 0]
    
    # Find min and max dates for each unique_id
    ds_bounds = non_null_data.groupby('unique_id')['ds'].agg(['min', 'max']).reset_index()
    ds_bounds.columns = ['unique_id', 'min_ds', 'max_ds']
    
    # Merge bounds to the original DataFrame
    merged_df = df.merge(ds_bounds, on='unique_id', how='left')
    
    # Filter using the bounds
    filtered_df = merged_df[(merged_df['ds'] >= merged_df['min_ds']) & 
                           (merged_df['ds'] <= merged_df['max_ds'])]
    
    # Remove temporary columns
    return filtered_df.drop(columns=['min_ds', 'max_ds'])

def prepare_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare time-related features in the DataFrame.
    
    Args:
        df: Input DataFrame with 'Country', 'Product', and 'Month' columns
        
    Returns:
        DataFrame with added 'unique_id' and 'ds' columns
    """
    df_copy = df.copy()
    df_copy['unique_id'] = df_copy['Country'] + '_' + df_copy['Product']
    df_copy['ds'] = pd.to_datetime(df_copy['Month'], format='%b%Y')
    return df_copy

def preprocess_ex1_final_sub(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data for final submission of exercise 1.
    
    Args:
        df: Input DataFrame with time series data
        
    Returns:
        Tuple of three DataFrames:
        - All zeros time series
        - Previously active time series
        - Currently active time series
    """
    # Prepare time features
    processed_df = prepare_time_features(df)
    
    # Get time series categorization
    null_id_list = get_null_time_series(processed_df)
    prev_id_active = get_ending_zero_series(processed_df, final_submission=True)
    
    # Split data into three categories
    all_zeros_data = processed_df[processed_df['unique_id'].isin(null_id_list)]
    
    prev_active_data = processed_df[processed_df['unique_id'].isin(prev_id_active)]
    prev_active_data = filter_by_active_period(prev_active_data)
    
    curr_active_data = processed_df[
        ~processed_df['unique_id'].isin(np.concatenate([null_id_list, prev_id_active]))
    ]
    curr_active_data = filter_by_active_period(curr_active_data)
    
    # Define column order for output
    column_order = ['unique_id', 'ds', 'Quantity', 'Country', 'Product']
    
    return (
        all_zeros_data[column_order], 
        prev_active_data[column_order], 
        curr_active_data[column_order]
    )

def preprocess_ex1(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data for exercise 1 with train-validation split.
    
    Args:
        df: Input DataFrame with time series data
        
    Returns:
        Tuple of four DataFrames:
        - All zeros time series (training)
        - Previously active time series (training)
        - Currently active time series (training)
        - Validation data
    """
    # Prepare time features
    processed_df = prepare_time_features(df)
    
    # Split into training and validation
    validation_df = processed_df[processed_df['ds'].dt.year == 2023].reset_index(drop=True)
    train_df = processed_df[processed_df['ds'].dt.year < 2023].reset_index(drop=True)
    
    # Get time series categorization
    null_id_list = get_null_time_series(train_df)
    prev_id_active = get_ending_zero_series(train_df, final_submission=False)
    
    # Split training data into three categories
    all_zeros_data = train_df[train_df['unique_id'].isin(null_id_list)]
    
    prev_active_data = train_df[train_df['unique_id'].isin(prev_id_active)]
    prev_active_data = filter_by_active_period(prev_active_data)
    
    curr_active_data = train_df[
        ~train_df['unique_id'].isin(np.concatenate([null_id_list, prev_id_active]))
    ]
    curr_active_data = filter_by_active_period(curr_active_data)
    
    # Define column order for output
    column_order = ['unique_id', 'ds', 'Quantity', 'Country', 'Product']
    
    return (
        all_zeros_data[column_order],
        prev_active_data[column_order],
        curr_active_data[column_order],
        validation_df[column_order]
    )