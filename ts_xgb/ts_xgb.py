from pdb import set_trace

import warnings
#warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

import statsmodels.api as sm

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.preprocessing import MinMaxScaler
    from pyramid.arima import auto_arima

    
def read_data():
    df = pd.read_csv('data/12.csv')
    df = df.loc[:, 'Date':'Y']
    df.set_index('Date', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    return df


def main():

    df = read_data()

    # Take logarithm of input data
    df = np.log(df)

    # Remove weekly seasonality
    ts_comps_f7 = sm.tsa.seasonal_decompose(df, model='add',
                                         two_sided=False, freq=7)
    weekly_seasonality = ts_comps_f7.seasonal
    ts_remainder = ts_comps_f7.observed - weekly_seasonality

    # Detrend data
    ts_comps_f365 = sm.tsa.seasonal_decompose(df, model='add',
                                              two_sided=False, freq=365)
    ts_trend = ts_comps_f365.trend
    ts_detrend = ts_comps_f365.observed - ts_trend
    ts_detrend.dropna(inplace=True)

    #ts_trend.plot()
    #ts_detrend.plot()
    #plt.show()

    # Model ts_detrend with XGBoost
    # Features
    # 1. Previous date sales
    # 2. Ordinal date (number of days since time series started)
    # 3. Day of month
    # 4. Day of week

    dataset = pd.DataFrame(index=ts_detrend.index)
    dataset['y'] = ts_detrend['Y']
    # 1.
    dataset['y_lag'] = dataset['y'].shift(1)
    # 2.
    dataset['time_vector'] = range(ts_detrend.shape[0])
    # 3.
    dataset['month_day'] = np.array([ix.timetuple().tm_mday for ix in dataset.index])
    # 4.
    dataset['week_day'] = np.array([ix.timetuple().tm_wday for ix in dataset.index])

    # Drop first row in dataset
    dataset.drop(dataset.index[0], inplace=True)
    
    # Target
    y = dataset['y']
    
    
    
    # Model ts_trend with ARIMA or LSTM
    
    set_trace()
    

if __name__ == "__main__":
    main()
    

'''
df = pd.read_csv('data/10_1191.csv')
df = df.loc[:, 'Date':'Y']
df.set_index('Date', inplace=True)
df.index = pd.DatetimeIndex(df.index)

components = sm.tsa.seasonal_decompose(df)
components.plot()
plt.show()
df_trend = components.trend.dropna()
df_seasonal = components.seasonal
df_resid = components.resid

# Forecast trend
arima_fit = auto_arima(df_trend, seasonal=False, start_p=0, start_q=0,
                       max_p=10, max_q=10, d=None, stepwise=True)
'''

