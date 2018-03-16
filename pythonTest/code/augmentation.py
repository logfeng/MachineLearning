import numpy as np
import pandas as pd
import csv
import copy
import random
import matplotlib.pyplot as plt

X = np.loadtxt("../data/train_data.csv", delimiter = ',', skiprows = 1)     # data
Y = np.loadtxt("../data/train_label.csv", delimiter = ',', skiprows = 1)    # label
Z = np.loadtxt("../data/train_classNum.csv", delimiter = ',', skiprows = 1) # classNum
length = X.shape[1]    # colume
mineral = X.shape[0]
augEach = 10           # shift number of spectrum in each class
classNum = Z.shape[0]
augData = copy.deepcopy(X)
augLabel = copy.deepcopy(Y)
i = 0                  # index
singleMinerIndex = []  # get rid of the data.

# augmentaion
# offset
# slope
# multiplication

for x, y in zip(X, Y) :
	# print x
	# print y
	y = int(y)
	if y % 500 == 0 :
		print("Class %d --------------------------------------------" % y)
	if Z[y] < 2 :
		singleMinerIndex = np.concatenate((singleMinerIndex, [i]))
	elif Z[y] >= augEach :
		pass
	else :
		shift_num = augEach - Z[y]
		shift_num = int(shift_num)
		# shift spectrum, shift each spectrum left or right a few wavenumbers randomly
		s2 = random.sample(range(1, 500+1), shift_num)
		for s in range(shift_num):
			augLeft = np.zeros((1, length))
			augRight = np.zeros((1, length))
			augLeft[0][0:(length - s2[s])] = x[s2[s]:length]
			augRight[0][s2[s]:length] = x[0:(length - s2[s])]
			augData = np.concatenate((augData, augRight), axis=0)
			augLabel = np.concatenate((Y, [y]))
	i = i + 1
# plt.figure()
# plt.plot(X[2566])
# plt.show()

augData = np.delete(augData, singleMinerIndex, axis=0)
augLabel = np.delete(augLabel, singleMinerIndex, axis=0)
data = pd.DataFrame(augData)
dataLabel = pd.DataFrame(augLabel)
minerIndex = pd.DataFrame(singleMinerIndex)
data.to_csv('../data/one/augmentation.csv', index = False)
dataLabel.to_csv('../data/one/augLabel.csv', index = False)
minerIndex.to_csv('../data/one/singleMinerIndex.csv', index = False)

# gauss random noise, proportional to the magnitude at each wave number
# different amplitude maybe need normalize
# mean, std = 0, 100
# s1 = np.random.normal(mean, std, length)
# augNoise = X[0] + s1

# plt.figure(num=2)
# plt.plot(X[0])
# plt.plot(augNoise)

# # linear combination
# s3 = np.random.randint(1,10,7)
# sum_matrix = np.sum(s3)
# proport = s3 / float(sum_matrix)
# # print (proport)
# augCombine = s3[0] * X[2] + s3[1] * X[3] + s3[2] * X[4] + s3[3] * X[5] +\
#              s3[4] * X[6] + s3[5] * X[7] + s3[6] * X[8]

# plt.figure(num=3)
# plt.plot(X[2])
# plt.plot(X[3])
# plt.plot(X[4])
# plt.plot(X[5])
# plt.plot(X[6])
# plt.plot(X[7])
# plt.plot(X[8])
# plt.plot(augCombine, color='red')
