import pandas as pd
#import matplotlib.pyplot as plt
from utils import preprocess
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM #DeepAR,NHITS,RNN,
import torch
from typing import Union
from neuralforecast.losses.pytorch import MSE
from neuralforecast.losses.pytorch import BasePointLoss,DistributionLoss,MQLoss
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM,DeepAR,StemGNN,TFT
from neuralforecast.utils import augment_calendar_df
from utils.losses import customLoss


df = pd.read_csv('data/01_input_history.csv')
df.head()


df_train_null, df_train_inactive, df_train_active, df_validation = preprocess.preprocess_ex1(df)


# merge the inactive to the active
# print(df_active.shape)
# print(df_inactive.shape)

df_train_merged = df_train_active#pd.merge(df_train_active, df_train_inactive, how='outer', on=['unique_id', 'ds', 'Quantity', 'Country', 'Product'])
df_train_static = df_train_merged[['unique_id', 'Country', 'Product']].drop_duplicates().reset_index(drop=True)
df_train_static = pd.get_dummies(df_train_static, columns=['Country', 'Product'], drop_first=True)
assert df_train_static.shape[0] == df_train_merged['unique_id'].nunique(), 'The number of unique_id in static and merged dataframes do not match!'

#df_train_merged = augment_calendar_df(df_train_merged, freq='M')[0]
#df_validation = augment_calendar_df(df_validation, freq='M')[0]
df_train_merged['month'] = df_train_merged['ds'].dt.month
df_validation['month'] = df_validation['ds'].dt.month
df_train_merged = pd.get_dummies(df_train_merged, columns=['month'], drop_first=True)
df_validation = pd.get_dummies(df_validation, columns=['month'], drop_first=True)

FORECASTING_HORIZON = 12 # one year ahead forecast

custom_loss = customLoss()

nf = NeuralForecast(
    models=[
        # Model 1: Long-short term memory
        # LSTM(
        #     h =FORECASTING_HORIZON,
        #     input_size= FORECASTING_HORIZON*2,
        #     loss=custom_loss,#DistributionLoss(distribution="NegativeBinomial"),
        #     encoder_n_layers = 5,
        #     encoder_hidden_size= 128,
        #     decoder_hidden_size=  64,
        #     decoder_layers= 2,
        #     encoder_dropout = 0.1,
        #     futr_exog_list = ['month_2','month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8','month_9', 'month_10', 'month_11', 'month_12'],#['month'],
        #     stat_exog_list = df_train_static.columns.tolist()[1:],
        #     batch_size = 1024,
        #     learning_rate= 6e-4,
        #     max_steps = 2500,
        #     scaler_type = None,#'robust',
        #     random_seed=1610,
        # )
        TFT(
            h=12,
            input_size=24,
            hidden_size=20,
            grn_activation="ELU",
            rnn_type="lstm",
            n_rnn_layers=1,
            one_rnn_initial_state=False,
            loss= MQLoss(),#DistributionLoss(distribution="NegativeBinomial"),# level=[80, 90]),
            learning_rate=0.005,
            futr_exog_list = ['month_2','month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8','month_9', 'month_10', 'month_11', 'month_12'],#['month'],
            stat_exog_list = df_train_static.columns.tolist()[1:],
            max_steps=5000,
            #val_check_steps=10,
            #early_stop_patience_steps=10,
            scaler_type="robust",
            #windows_batch_size=None,
            #enable_progress_bar=True,
            batch_size=1024,
        ),
    ],
    freq='MS',

)

nf.fit(
    #df = df_train_merged[['unique_id','ds','Quantity',  'month']],
    df = df_train_merged.drop(columns=['Country', 'Product']),
    static_df= df_train_static,
    #val_size= 12
    # id_col = 'unique_id'
    target_col = 'Quantity'
)

future_df = nf.make_future_dataframe()

#future_df = augment_calendar_df(future_df, freq='M')[0]
future_df['month'] = future_df['ds'].dt.month
future_df = pd.get_dummies(future_df, columns=['month'], drop_first=True)


y_hat = nf.predict(
    futr_df = future_df,
)

y_hat=y_hat.loc[y_hat['unique_id'].isin(df_train_active['unique_id'].unique()),:]

df_forecast = df_validation[['unique_id', 'ds']].copy()
df_forecast['Quantity'] = 0

for id in y_hat['unique_id'].unique():
    df_forecast.loc[df_forecast['unique_id'] == id, 'Quantity'] = y_hat.loc[y_hat['unique_id'] == id, 'TFT'].values.astype(int)
    
def restore_original_format(date_column):
    return date_column.dt.strftime('%b%Y')

def submission_formatter(df):
    restored_df = df.copy()
    restored_df['Country'] = df['unique_id'].str.split('_').str[0]
    restored_df['Product'] = df['unique_id'].str.split('_').str[1]
    restored_df['Month'] = restore_original_format(restored_df['ds'])
    restored_df.drop(columns=['unique_id', 'ds'], inplace=True)
    return restored_df

    
df_forecast = submission_formatter(df_forecast)
df_validation = submission_formatter(df_validation)
# df_validation.drop(columns=['month'], inplace=True)

# # save to csv
df_forecast.to_csv('submissions/submission_TFT.csv', index=False)
df_validation.to_csv('submissions/validation.csv', index=False)