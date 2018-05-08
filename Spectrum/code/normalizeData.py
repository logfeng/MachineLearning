# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

X = np.loadtxt("../data/train_data_select.csv", delimiter = ',', skiprows = 1, encoding='utf-8')     # data
X = X.T
scaler = MinMaxScaler()
scaler.fit(X)
scaler.data_max_
X_normalize = scaler.transform(X)
X_normalize = X_normalize.T
X_normalize = pd.DataFrame(X_normalize)
X_normalize.to_csv('../data/train_data_normalize.csv', index = False)

# X = np.loadtxt("../data/train_data_normalize.csv", delimiter = ',', skiprows = 1, encoding='utf-8')     # data
# print(X[11,:])
