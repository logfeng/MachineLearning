import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt("../data/train_data.csv", delimiter = ',', skiprows = 1)  # data
Y = np.loadtxt("../data/train_label.csv", delimiter = ',', skiprows = 1) # label
length = X.shape[1]  # colume

# augmentaion
# offset
# slope
# multiplication

# shift spectrum, shift each spectrum left or right a few wavenumbers randomly
shift_range = 100
s2 = np.random.randint(1, shift_range, 100)
print ('s2: ',s2)
augLeft = np.zeros(length)
augRight = np.zeros(length)
augLeft[0:(length - s2[0])] = X[0][s2[0]:length]
augRight[s2[0]:length] = X[0][0:(length - s2[0])]

plt.figure()
plt.plot(X[0])
plt.plot(augLeft)
plt.plot(augRight)

# gauss random noise, proportional to the magnitude at each wave number
mean, std = 0, 100
s1 = np.random.normal(mean, std, length)
augNoise = X[0] + s1

plt.figure(num=2)
plt.plot(X[0])
plt.plot(augNoise)

# linear combination
s3 = np.random.randint(1,10,7)
sum_matrix = np.sum(s3)
proport = s3 / float(sum_matrix)
print (proport)
augCombine = s3[0] * X[2] + s3[1] * X[3] + s3[2] * X[4] + s3[3] * X[5] +\
             s3[4] * X[6] + s3[5] * X[7] + s3[6] * X[8]

plt.figure(num=3)
plt.plot(X[2])
plt.plot(X[3])
plt.plot(X[4])
plt.plot(X[5])
plt.plot(X[6])
plt.plot(X[7])
plt.plot(X[8])
plt.plot(augCombine, color='red')
plt.show()
