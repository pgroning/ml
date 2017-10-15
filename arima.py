#!/usr/bin/python

from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt
 
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
print(series.head())
plt.figure(1)
series.plot()

plt.figure(2)
autocorrelation_plot(series)

# Positive correlation for ~10 lags.
# The plot shows that the number of significant lags ~5 and therefore the AR paramter should be ~5

# Fit ARIMA(5,1,0)
# This sets the lag value to 5 for autoregression, using difference order of 1 and no moving average model.
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
residuals.plot(kind='kde')

plt.show()
