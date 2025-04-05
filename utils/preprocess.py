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

def end_zero(df:pd.DataFrame):
    last_null_shipment = df[(df['Month'] == 'Dec2023') & (df['Quantity'] == 0)]
    df_grouped = count_quantity(last_null_shipment)
    return df_grouped[df_grouped['Quantity'] != 0]['unique_id'].values

def preprocess_ex1(df):
    df['unique_id'] = df['Country'] + '_' + df['Product']
    df['ds'] = pd.to_datetime(df['Month'], format='%b%Y')
    null_id_list = null_ts(df)
    prev_id_active = end_zero(df)

    # Data 1: All zeros
    all_zeros_data = df[df['unique_id'].isin(null_id_list)]
    
    # Data 2: Previously active series
    prev_active_data = df[(df['unique_id'].isin(prev_id_active)) & (df['Quantity'] != 0)]

    # Data 3: Currently active
    curr_active_data = df[(~df['unique_id'].isin(np.concatenate([null_id_list, prev_id_active]))) & \
                          (df['Quantity'] != 0)]

    column_order = ['unique_id', 'ds', 'Quantity', 'Country', 'Product']

    return all_zeros_data[column_order], prev_active_data[column_order], curr_active_data[column_order]