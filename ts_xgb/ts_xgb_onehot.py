from pdb import set_trace

import warnings
#warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import scipy.stats as st

import statsmodels.api as sm

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.preprocessing import MinMaxScaler
    from pyramid.arima import auto_arima

    
def read_data():
    df = pd.read_csv('data/12.csv')
    #df = pd.read_csv('data/10_1191.csv')
    df = df.loc[:, 'Date':'Y']
    df.set_index('Date', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    return df


def fourier(t, k=1, m=365.25):
    x = np.zeros((len(t), 2 * k))
    i = 0
    for j in range(1, k + 1):
        x[:, i] = np.cos(2 * np.pi * j / m * t)
        x[:, i + 1] = np.sin(2 * np.pi * j / m * t)
        i += 2    
    return x

def xgb_dataset(y_series, periods):
    # Create feature dataset for XGBoost
    # Features
    # 1. Ordinal date (number of days since time series started)
    # 2. Yearly seasonality (expressed as fourier terms m=365.25)
    # 3. Day of week (expressed as one-hot-encoding)
    
    dataset = pd.DataFrame(index=y_series.index)
    dataset['y'] = y_series

    if periods > 0:
        forecast_dates = pd.date_range(start=dataset.index[-1], periods=periods+1)[1:]
        df_forecast = pd.DataFrame(index=forecast_dates)
        df_forecast['y'] = np.nan
        dataset = pd.concat([dataset, df_forecast])
    else:
        periods = -periods
        
    t = np.array([i for i in range(dataset.shape[0])])
    # 1.
    dataset['time'] = t
    # 2.
    ft = fourier(t, m=365.25, k=2)
    dataset['C1'] = ft[:, 0]
    dataset['S1'] = ft[:, 1]
    dataset['C2'] = ft[:, 2]
    dataset['S2'] = ft[:, 3]
    # 3.
    week_days = np.array([ix.timetuple().tm_wday for ix in dataset.index])
    wday_ohe = OneHotEncoder(categories='auto')
    week_days_reshape = week_days.reshape(-1, 1)
    week_days_ohe = wday_ohe.fit_transform(week_days_reshape).toarray()
    df_wd_ohe = pd.DataFrame(week_days_ohe, index=dataset.index)
    dataset = pd.concat([dataset, df_wd_ohe], axis=1)

    df_historic = dataset.iloc[:-periods, :]
    df_forecast = dataset.iloc[-periods:, :]
    
    return df_historic, df_forecast


def xgb_forecast(df_train, df_forecast):
    # Forecasting using XGBoost

    # Train model & tune hyperparameters

    one_to_left = st.beta(10, 1)
    from_zero_positive = st.expon(0, 50)
    
    params = {
        "n_estimators": st.randint(10, 20000),
        "max_depth": st.randint(1, 4),
        "learning_rate": st.uniform(0.01, 0.09),
        #"colsample_bytree": one_to_left,
        #"subsample": one_to_left,
        #"gamma": st.uniform(0, 10),
        #"reg_alpha": from_zero_positive,
        #"min_child_weight": from_zero_positive,
    }

    X_train = df_train.drop('y', axis=1).values
    y_train = df_train['y'].values
    
    xgbreg = XGBRegressor(nthreads=-1, booster='gbtree')
    gs = RandomizedSearchCV(xgbreg, params, n_jobs=1)
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    print(model)

    # Create forecast
    X_forecast = df_forecast.drop('y', axis=1).values
    y_hat = model.predict(X_forecast)
    df_yhat = pd.DataFrame(y_hat, index=df_forecast.index, columns=['yhat'])

    return df_yhat
    

def main():

    df = read_data()

    # Take logarithm of input data
    df = np.log(df)

    # Remove trend from data
    y_comps = sm.tsa.seasonal_decompose(df['Y'], model='add', two_sided=False, freq=365)
    y_trend = y_comps.trend.dropna()
    y_trend_split = sm.tsa.seasonal_decompose(y_trend, model='add', two_sided=False, freq=7)

    ts_trend = y_trend_split.trend
    ts_remainder = y_comps.observed - ts_trend

    ts_trend.dropna(inplace=True)
    ts_remainder.dropna(inplace=True)

    
    # Forecast detrended time series with XGBoost
    
    periods = 365
    df_hist, df_forecast = xgb_dataset(ts_remainder, periods=periods)
    
    yhat = xgb_forecast(df_hist, df_forecast)
    
    # Plot results
    #ax = df_hist['y'].plot(c='b')
    ax = ts_remainder.plot(c='b')
    yhat.plot(ax=ax, c='r')
    plt.show()

    
    # Forecast trend with ARIMA or LSTM


    
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

