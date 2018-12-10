from pdb import set_trace

import warnings
#warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
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

    #ts_detrend_f365 = y_comps_f365.observed - y_comps_f365.trend
    #ts_detrend_f365.dropna(inplace=True)
    #y_comps_f7 = sm.tsa.seasonal_decompose(ts_detrend_f365, model='add', two_sided=False, freq=7)

    #ts_comps_f7 = sm.tsa.seasonal_decompose(df, model='add',
    #                                        two_sided=False, freq=7)
    #weekly_seasonality = ts_comps_f7.seasonal
    #ts_remainder = ts_comps_f7.observed - weekly_seasonality

    ## Detrend data
    #ts_comps_f365 = sm.tsa.seasonal_decompose(ts_remainder, model='add',
    #                                          two_sided=False, freq=365)
    #yearly_seasonality = ts_comps_f365.seasonal
    #ts_trend = ts_comps_f365.trend
    #ts_detrend = ts_comps_f365.observed - ts_trend
    #ts_detrend.dropna(inplace=True)

    #ts_trend.plot()
    #ts_detrend.plot()
    #plt.show()

    #trend_comps = sm.tsa.seasonal_decompose(ts_trend.dropna(), model='add', two_sided=False, freq=7)

    # Model ts_detrend with XGBoost
    #ts_detrend_comps = sm.tsa.seasonal_decompose(ts_detrend, model='add', two_sided=False, freq=365)
    #ts_detrend = ts_detrend_comps.observed - ts_detrend_comps.trend
    #ts_detrend.dropna(inplace=True)
    
    # Features
    # 1. Previous date sales
    # 2. Ordinal date (number of days since time series started)
    # 3. Day of month
    # 4. Day of week

    dataset = pd.DataFrame(index=ts_remainder.index)
    dataset['y'] = ts_remainder
    # 1.
    dataset['y_lag'] = dataset['y'].shift(1)
    # 2.
    dataset['time_vector'] = range(ts_remainder.shape[0])
    # 3.
    dataset['month_day'] = np.array([ix.timetuple().tm_mday for ix in dataset.index])
    # 4.
    dataset['week_day'] = np.array([ix.timetuple().tm_wday for ix in dataset.index])
    # 5.
    dataset['month'] = np.array([ix.timetuple().tm_mon for ix in dataset.index])
    
    # Drop first row in dataset
    dataset.drop(dataset.index[0], inplace=True)

    # Split data in train and test sets
    n_test = 7
    train_X = dataset.iloc[:-n_test, 1:]
    train_y = dataset.iloc[:-n_test, 0]
    test_X = dataset.iloc[-n_test:, 1:]
    test_y = dataset.iloc[-n_test:, 0]

    # Train model
    params = {
        'objective': 'reg:linear',
        'max_depth': 2,
        'learning_rate': 0.05,
        'silent': 1.0,
        'n_estimators': 10000
    }

    #xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, booster='gbtree', max_depth=3)
    xgb_model = XGBRegressor(**params)
    xgb_model.fit(train_X.values, train_y.values)

    #xgb.plot_tree(xgb_model, num_trees=4, rankdir='LR')
        
    y_hat = xgb_model.predict(test_X.values)
    print('y_hat = ', y_hat)
    print('y = ', test_y)
    #xgb_model.feature_importances_
    xgb.plot_importance(xgb_model)
    plt.show()

    # Hyperparameter Tuning
    one_to_left = st.beta(10, 1)
    from_zero_positive = st.expon(0, 50)
    
    params = {
        "n_estimators": st.randint(10, 10000),
        "max_depth": st.randint(1, 4),
        "learning_rate": st.uniform(0.01, 0.5),
        "colsample_bytree": one_to_left,
        "subsample": one_to_left,
        "gamma": st.uniform(0, 10),
        "reg_alpha": from_zero_positive,
        "min_child_weight": from_zero_positive,
    }

    xgbreg = XGBRegressor(nthreads=-1)
    gs = RandomizedSearchCV(xgbreg, params, n_jobs=1)
    gs.fit(train_X.values, train_y.values)
    print(gs.best_estimator_)
    y_hat = xgb_model.predict(test_X.values)
    print('y_hat_opt = ', y_hat)

    
    #max_depth = [1, 2, 3, 4]
    #learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    #param_grid = dict(learning_rate=learning_rate)
    #param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
    #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    #grid_search = GridSearchCV(xgb_model, param_grid, n_jobs=-1, cv=kfold)
    #grid_search = GridSearchCV(xgb_model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    #grid_result = grid_search.fit(train_X.values, train_y.values)
    
    
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

