from pdb import set_trace

import warnings
#warnings.filterwarnings("ignore")

import numpy as np
from numpy import array
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Bidirectional
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.optimizers import SGD
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


def decompose(df, model='additive'):
    ts_components = sm.tsa.seasonal_decompose(df, model)
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


def split_sequence(sequence, n_steps_in, n_steps_out):

    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        #out_end_ix = end_ix + n_steps_out
        # Check if we have reached the end of sequence
        if end_ix > len(sequence):
            break
        #if out_end_ix > len(sequence):
        #    break
        
        seq_x = sequence[i:end_ix, :-1]
        seq_y = sequence[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)

    return array(X), array(y)

    
def build_model(n_steps_in, n_steps_out, n_features, n_units):

    n_layers = len(n_units)
    
    model = Sequential()
    if n_layers == 1:
        model.add(LSTM(n_units[0], activation='relu',
                      input_shape=(n_steps_in, n_features)))
        #layerBi = Bidirectional(layer1)
        #model.add(layerBi)
    else:
        for i in range(n_layers - 1):
            model.add(LSTM(n_units[i], activation='relu', return_sequences=True,
                           input_shape=(n_steps_in, n_features)))
            model.add(Dropout(0.2))
        model.add(LSTM(n_units[-1], activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(units=n_steps_out))
    model.add(Activation('linear'))
    
    #optimizer = SGD(lr=0.5, decay=1e-6, momentum=0.1, nesterov=True)
    #model.compile(optimizer=optimizer, loss='mse')
    model.compile(optimizer='adam', loss='mse')

    return model

    
def main():

    #df = read_data()
    #ts_components = decompose(df, model='multiplicative')
    #set_trace()
    #trend_forecast = forecast_trend(ts_components.trend.dropna())
    #y = subtract_trend(ts_components, trend_forecast)
    #y.to_csv('y_subtract.csv')

    #df = pd.read_csv('12_trend_multi.csv')
    df = pd.read_csv('12_seas_resid_multi.csv')
    
    df.set_index('Date', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    #df.plot()
    #plt.show()

    # log transform data
    #df['Y'] = np.log(df['Y'])
    
    # Scaling data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(df).flatten()

    # Adding regressors
    xog = np.linspace(0, 1, len(data))

    # shift data one step
    data_shift = np.empty(len(data))
    data_shift[:-1] = data[1:]
    dataset = np.hstack((data.reshape(len(data), 1), xog.reshape(len(xog), 1), data_shift.reshape(len(data_shift), 1)))
    dataset = dataset[:-1, :]
    
    # Prepare univariate data for modeling
    n_steps_in = 1
    n_steps_out = 1
    X, y = split_sequence(dataset, n_steps_in, n_steps_out)

    #for i in range(len(X)):
    #    print(X[i], y[i])
    
    # Setup model
    n_features = X.shape[2]
    model = build_model(n_steps_in, n_steps_out, n_features,
                        n_units=[100, 400])
    
    # Train model
    model.fit(X, y, epochs=1000, batch_size=400, verbose=1)
    
    # Make single step prediction
    x_input = dataset[-n_steps_in:, :-1]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    y_hat = model.predict(x_input, verbose=0)
    print('----')
    print('y_hat = ', scaler.inverse_transform(y_hat).flatten())
    print('y = ', df['Y'][-n_steps_in:])
    
    # Store forecast in Pandas dataframe
    Y_hat = scaler.inverse_transform(y_hat).flatten()
    dt_range = pd.date_range(df['Y'].index[-1], periods=n_steps_out+1)[1:]
    df_yhat = pd.DataFrame(Y_hat.transpose(), columns=['Y_hat'], index=dt_range)

    # Plot results
    #ax = df[:].plot(c='b')
    #df_yhat.plot(c='r', ax=ax)
    #plt.show()
    set_trace()
    

if __name__ == "__main__":
    main()
    
