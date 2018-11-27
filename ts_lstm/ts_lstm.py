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

df = pd.read_csv('../ts_xgb/data/12.csv')
df = df.loc[:, 'Date':'Y']
df.set_index('Date', inplace=True)
df.index = pd.DatetimeIndex(df.index)

ts_components = sm.tsa.seasonal_decompose(df)
ts_components.plot()
plt.show()
df_trend = ts_components.trend.dropna()

# Forecast trend with Arima
arima_model = auto_arima(df_trend, seasonal=False, start_p=0, start_q=0,
                       max_p=10, max_q=10, d=None, error_action='ignore',
                       suppress_warnings=True, stepwise=True)
print("arima aic=", arima_model.aic())

# Train arima model
arima_model.fit(df_trend)
# Forecast trend
trend_forecast = arima_model.predict(n_periods=33)
ts_components.trend['Y'].iloc[-3:] = trend_forecast[:3]
ts_components.trend.plot()
plt.show()
set_trace()
