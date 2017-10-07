from pdb import set_trace
import datetime
import pandas as pd


idx = pd.date_range("2015-01-01", "2017-09-21", freq="D")
df = pd.DataFrame(index=idx, columns=['quantity'])

ip = df.index.get_loc("2017-08-08", method="nearest")
wday = df.index[ip].weekday()

#dateoffset = pd.DateOffset(365.25)
ip = df.index.get_loc("2016-08-08", method="nearest")
days_ahead = df.index[ip].weekday() - wday
ip -= days_ahead


set_trace()
