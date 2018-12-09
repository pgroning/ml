from pdb import set_trace

import warnings
#warnings.filterwarnings("ignore")

import numpy as np
from numpy import array
import pandas as pd
from datetime import datetime
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


def decompose(df, model='multiplicative', frequency=7):
    ts_components = sm.tsa.seasonal_decompose(df, model, freq=frequency, two_sided=False)
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
        out_end_ix = end_ix + n_steps_out - 1
        # Check if we have reached the end of sequence
        if out_end_ix > len(sequence):
            break
        
        seq_x = sequence[i:end_ix, :-1]
        seq_y = sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)

    return array(X), array(y)

    
def build_model(n_steps_in, n_steps_out, n_features, n_units, bidir=True, dropout=0.2):

    n_layers = len(n_units)
    
    model = Sequential()
    if n_layers == 1:
        layer = LSTM(n_units[0], activation='relu', input_shape=(n_steps_in, n_features))
        if bidir:
            layer = Bidirectional(layer)
        model.add(layer)
    else:
        layers = list()
        for i in range(n_layers - 1):
            layers.append(LSTM(n_units[i], activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
            if bidir:
                layers[-1] = Bidirectional(layers[-1])
            model.add(layers[-1])
            model.add(Dropout(dropout))

        layers.append(LSTM(n_units[-1], activation='relu'))
        if bidir:
            layers[-1] = Bidirectional(layers[-1])
        model.add(layers[-1])

    model.add(Dropout(dropout))
    model.add(Dense(units=n_steps_out))
    model.add(Activation('linear'))
    
    #optimizer = SGD(lr=0.5, decay=1e-6, momentum=0.1, nesterov=True)
    #model.compile(optimizer=optimizer, loss='mse')
    model.compile(optimizer='adam', loss='mse')

    return model


def fourier(t, k=1, m=365.25):
    x = np.zeros((len(t), 2 * k))
    i = 0
    for j in range(1, k + 1):
        x[:, i] = np.cos(2 * np.pi * j / m * t)
        x[:, i + 1] = np.sin(2 * np.pi * j / m * t)
        i += 2
    
    return x


def main():

    #df = read_data()
    #ts_components = decompose(df, model='multiplicative', frequency=7)

    #weekly_removed = ts_components.observed / ts_components.seasonal
    #ts_components = decompose(weekly_removed, model='multiplicative', frequency=365)

    #yearly_removed = ts_components.observed / ts_components.seasonal

    #set_trace()
    #trend_forecast = forecast_trend(ts_components.trend.dropna())
    #y = subtract_trend(ts_components, trend_forecast)
    #y.to_csv('y_subtract.csv')
    
    df = pd.read_csv('12_yw_removed.csv')
    #df = pd.read_csv('y_observed.csv')
    #df = pd.read_csv('12_trend_multi.csv')
    #df = pd.read_csv('12_seas_resid_multi.csv')
    #df = pd.read_csv('12_trend_resid_multi.csv')
    
    df.set_index('Date', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    #df.plot()
    #plt.show()

    # log transform data
    df['Y'] = np.log(df['Y'])
    
    # Scaling data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(df).flatten()

    # Adding regressors
    # harmonics
    t = np.linspace(0, 1, len(data))
    ft = fourier(t, k=1)
    xfourier = list()
    for i in range(ft.shape[1]):
        xfourier.append(ft[:, i].reshape(len(ft[:, i]), 1))

    ft7 = fourier(t, k=1, m=7)
    xfourier7 = list()
    for i in range(ft7.shape[1]):
        xfourier7.append(ft7[:, i].reshape(len(ft7[:, i]), 1))
        
    # time-trend
    time_trend = t.reshape(len(t), 1)

    # day-of-year
    yday = np.array([ix.timetuple().tm_yday for ix in df.index])
    yday = yday / yday.max()
    year_day = yday.reshape(len(yday), 1)
    
    # day-of-week
    wday = np.array([ix.timetuple().tm_wday for ix in df.index])
    wday = wday / wday.max()
    week_day = wday.reshape(len(wday), 1)
    
    # shift data one steps
    data_shift = np.empty(len(data))
    data_shift[:-1] = data[1:]
    y_shift = data_shift.reshape(len(data_shift), 1)

    # Prepare multivariate data for modeling
    x = data.reshape(len(data), 1)

    st = list()
    #st = [x for x in xfourier]
    #for xf7 in xfourier7:
    #    st.append(xf7)
    
    st.append(time_trend)
    #st.append(year_day)
    #st.append(week_day)
    st.append(x)
    st.append(y_shift)
    dataset = np.hstack(st)
    dataset_train = dataset[:-1, :] # Drop last datapoint
    
    # Prepare multivariate data for modeling
    n_steps_in = 30
    n_steps_out = 30
    X, y = split_sequence(dataset_train, n_steps_in, n_steps_out)

    #for i in range(len(X)):
    #    print(X[i], y[i])
    #set_trace()
        
    # Setup model
    # 20
    n_features = X.shape[2]
    model = build_model(n_steps_in, n_steps_out, n_features,
                        n_units=[20], bidir=True, dropout=0.2)
    
    # Train model
    model.fit(X, y, epochs=3000, batch_size=730, verbose=1)
    
    # Make single step prediction
    x_input = dataset[-n_steps_in:, :-1]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    set_trace()
    y_hat = model.predict(x_input, verbose=0)
    print('----')
    print('y_hat = ', scaler.inverse_transform(y_hat).flatten())
    print('y = ', df['Y'][-n_steps_in:])
    
    # Store forecast in Pandas dataframe
    Y_hat = scaler.inverse_transform(y_hat).flatten()
    dt_range = pd.date_range(df['Y'].index[-1], periods=n_steps_out+1)[1:]
    df_yhat = pd.DataFrame(Y_hat.transpose(), columns=['Y_hat'], index=dt_range)

    # Plot results
    df['Y'] = np.exp(df['Y'])
    df_yhat = np.exp(df_yhat)
    
    ax = df[:].plot(c='b')
    df_yhat.plot(c='r', ax=ax)
    plt.show()
    set_trace()
    

if __name__ == "__main__":
    main()
    
