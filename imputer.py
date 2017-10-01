import numpy as np
from sklearn.preprocessing import Imputer

X = np.array([[1,1,np.nan],[2,np.nan,2],[np.nan,3,3]])
print X

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X)
Y = imputer.transform(X)
print Y


X2 = np.array([[1,1,0],[2,0,2],[0,3,3]])
print X2

imputer = Imputer(missing_values=0, strategy='median', axis=0)
imputer = imputer.fit(X2)
Y2 = imputer.transform(X2)
print Y2
