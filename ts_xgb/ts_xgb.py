from pdb import set_trace

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import pandas as pd

import statsmodels.api as sm
from pyramid.arima import auto_arima


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

set_trace()
