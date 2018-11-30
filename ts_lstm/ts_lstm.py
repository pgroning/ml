from pdb import set_trace

import warnings
#warnings.filterwarnings("ignore")

from numpy import array
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt

import statsmodels.api as sm

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.preprocessing import MinMaxScaler
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


def split_sequence(sequence, n_steps):

    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        # Check if we have reached the end of sequence
        if end_ix > len(sequence) - 1:
            break

        seq_x = sequence[i:end_ix]
        seq_y = sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    return array(X), array(y)
    

def single_lstm(n_steps, n_features, n_units=50):

    model = Sequential()
    layer1 = LSTM(n_units, activation='relu', return_sequences=True, input_shape=(n_steps, n_features))
    model.add(layer1)
    model.add(LSTM(n_units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model

    
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

    # Scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(df)
    
    # Prepare univariate data for modeling
    n_steps = 100
    #n_steps = 7*8  # 8 weeks
    #X, y = split_sequence(array(df['Y']), n_steps)
    X, y = split_sequence(data, n_steps)
    
    # Setup model
    n_features = 1
    model = single_lstm(n_steps, n_features, n_units=50)

    # Add dimensions for features
    X_reshaped = X.reshape((X.shape[0], X.shape[1], n_features))

    # Train model
    model.fit(X_reshaped, y, epochs=20, verbose=1)
    #model.fit(X_reshaped, y, epochs=100, batch_size=1, verbose=1)

    # Make single step prediction
    x_input = data[-n_steps-1:-1]
    x_input = x_input.reshape((1, n_steps, n_features))
    #x_input = array(df['Y'][-n_steps-1:-1])
    #x_input = x_input.reshape((1, n_steps, n_features))
    y_hat = model.predict(x_input, verbose=0)
    print('y_hat=', y_hat)
    print(data[-7:])
    print('----')
    print('y_hat=', scaler.inverse_transform(y_hat))
    #print(df['Y'][-n_steps-1:])
    print('y=', df['Y'][-1])
    set_trace()
    

if __name__ == "__main__":
    main()
    
