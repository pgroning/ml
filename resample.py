from pdb import set_trace
import datetime
import pandas as pd
import numpy as np


idx = pd.date_range("2015-01-01", "2015-12-31", freq="D")
df = pd.DataFrame(index=idx, columns=['quantity'])
df['quantity'] = np.array(range(len(df)),dtype='float32')/len(df)
df_w = df.resample('W-MON', how='mean')
df_mean = df_w.resample('D', fill_method='bfill')

i_end = df.index[-1]
I = df / df_mean[:str(i_end)]

set_trace()
