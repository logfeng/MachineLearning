import numpy as np

X = np.loadtxt("train_data.csv", delimiter = ',', skiprows = 1)  # data
Y = np.loadtxt("train_label.csv", delimiter = ',', skiprows = 1) # label
