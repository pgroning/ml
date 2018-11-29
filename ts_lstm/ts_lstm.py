from pdb import set_trace

import warnings
warnings.filterwarnings("ignore")

from numpy import array
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt

import statsmodels.api as sm
from pyramid.arima import auto_arima

def read_data():
    df = pd.read_csv('../ts_xgb/data/12.csv')
    df = df.loc[:, 'Date':'Y']
    df.set_index('Date', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    return df


def decompose(df):
    ts_components = sm.tsa.seasonal_decompose(df)
    ts_components.plot()
    plt.show()
    return ts_components


def forecast_trend(df_trend):
    #df_trend = ts_components.trend.dropna()
    
    # Forecast trend with Arima
    arima_model = auto_arima(df_trend, seasonal=False, start_p=0, start_q=0,
                             max_p=10, max_q=10, d=None, error_action='ignore',
                             suppress_warnings=True, stepwise=True)
    print("arima aic=", arima_model.aic())

    # Train arima model
    arima_model.fit(df_trend)
    
    # Forecast trend
    trend_forecast = arima_model.predict(n_periods=33)
    return trend_forecast


def subtract_trend(ts_components, trend_forecast):

    ts_components.trend['Y'].iloc[-3:] = trend_forecast[:3]
    
    # Remove trend from observed data
    y_diff = ts_components.observed - ts_components.trend
    y_diff.dropna(inplace=True)
    y_diff.plot()
    plt.show()
    return y_diff


def main():

    #df = read_data()
    #ts_components = decompose(df)
    #trend_forecast = forecast_trend(ts_components.trend.dropna())
    #y = subtract_trend(ts_components, trend_forecast)
    #y.to_csv('y_subtract.csv')

    df = pd.read_csv('y_subtract.csv')
    df.set_index('Date', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    #df.plot()
    #plt.show()

    
    
    set_trace()
    

if __name__ == "__main__":
    main()
    
