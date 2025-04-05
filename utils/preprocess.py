import pandas as pd
import numpy as np

def count_quantity(df:pd.DataFrame):
    df_grouped = df.groupby('unique_id').agg({
        'Quantity':'sum'
    }).reset_index()

    return df_grouped

def null_ts(df):
    df_grouped = count_quantity(df)
    return df_grouped[df_grouped['Quantity'] == 0]['unique_id'].values

def end_zero(df:pd.DataFrame, final_sub=True):
    null_list = null_ts(df)
    last_year = 2023 if final_sub else 2022
    return df[(~df['unique_id'].isin(null_list)) & (df['ds'].dt.year == last_year) & (df['Quantity'] == 0)]['unique_id'].values

def filter_ds(df):
    # Prendi solo righe con Quantity != 0
    non_null_data = df[df['Quantity'] != 0]

    # Trova min e max ds per ogni unique_id
    ds_bounds = non_null_data.groupby('unique_id')['ds'].agg(['min', 'max']).reset_index()
    ds_bounds.columns = ['unique_id', 'min_ds', 'max_ds']

    # Unisci i bounds al DataFrame originale
    df = df.merge(ds_bounds, on='unique_id', how='left')

    # Filtra il df usando i bounds
    df_filtered = df[(df['ds'] >= df['min_ds']) & (df['ds'] <= df['max_ds'])]

    # Rimuovi le colonne temporanee
    return df_filtered.drop(columns=['min_ds', 'max_ds'])

def preprocess_ex1_final_sub(df):
    df['unique_id'] = df['Country'] + '_' + df['Product']
    df['ds'] = pd.to_datetime(df['Month'], format='%b%Y')
    null_id_list = null_ts(df)
    prev_id_active = end_zero(df, final_sub=True)

    # Data 1: All zeros
    all_zeros_data = df[df['unique_id'].isin(null_id_list)]
    
    # Data 2: Previously active series
    prev_active_data = df[(df['unique_id'].isin(prev_id_active))]
    prev_active_data = filter_ds(prev_active_data)

    # Data 3: Currently active
    curr_active_data = df[(~df['unique_id'].isin(np.concatenate([null_id_list, prev_id_active])))]
    curr_active_data = filter_ds(curr_active_data)

    column_order = ['unique_id', 'ds', 'Quantity', 'Country', 'Product']

    return all_zeros_data[column_order], prev_active_data[column_order], curr_active_data[column_order]


def preprocess_ex1(df):
    df['unique_id'] = df['Country'] + '_' + df['Product']
    df['ds'] = pd.to_datetime(df['Month'], format='%b%Y')

    validation_df = df[df['ds'].dt.year == 2023]
    train_df = df[df['ds'].dt.year < 2023]


    null_id_list = null_ts(train_df)
    prev_id_active = end_zero(train_df, final_sub=False)
    # Data 1: All zeros
    all_zeros_data = train_df[train_df['unique_id'].isin(null_id_list)]
    
    # Data 2: Previously active series
    prev_active_data = train_df[(train_df['unique_id'].isin(prev_id_active))]
    prev_active_data = filter_ds(prev_active_data)

    # Data 3: Currently active
    curr_active_data = train_df[(~train_df['unique_id'].isin(np.concatenate([null_id_list, prev_id_active])))]
    curr_active_data = filter_ds(curr_active_data)

    column_order = ['unique_id', 'ds', 'Quantity', 'Country', 'Product']

    return all_zeros_data[column_order], prev_active_data[column_order], curr_active_data[column_order], validation_df