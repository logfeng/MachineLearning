# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import pandas as pd
from scipy import interpolate
import math
import matplotlib.pyplot as plt

# filepath = '../data/one/'
filepath = '../data/excellent_unoriented/'
files = os.listdir(filepath)
pieces = []
specify_str = '532'
maxSpectrum = 1800
minSpectrum = 85
numSpectrum = maxSpectrum - minSpectrum + 1
speciesTmp = ''
label = 0
classNum = 1
# preprocess, resample
for file in files:
	if os.path.splitext(file)[1] == '.txt' and specify_str in file:
		print ("------Load " + file + "------")
		dataPath = filepath + file
		species = file.split('__')[0]
		# print (species == speciesTmp)
		if species == speciesTmp :
			classNum = classNum + 1
			# pass
		else :
			label = label + 1
			classNum = 1
		# print (label)
		speciesTmp = species
		txt = np.loadtxt(dataPath, delimiter = ', ')
		# print txt
		txtDF = pd.DataFrame(txt, columns = list('AB')) # 用波数代替columns list('AB')
		# print txtDF
		x = txtDF['A'].T
		y = txtDF['B'].T
		xmax = math.floor(max(x))
		xmin = math.ceil(min(x))
		if (xmin < minSpectrum) :
			xmin = minSpectrum
		if (xmax > maxSpectrum) :
			xmax = maxSpectrum
		xnum = xmax - xmin + 1
		xnew = np.linspace(xmin, xmax, xnum)
		f = interpolate.interp1d(x,y,kind = 'cubic')
		ynew = f(xnew)

		xres = np.linspace(minSpectrum, maxSpectrum, numSpectrum)
		xmin = int(xmin)
		xmax = int(xmax)
		res1 = list(np.zeros(xmin - minSpectrum))
		res2 = list(np.zeros(maxSpectrum - xmax))
		resample = res1 + list(ynew) + res2

		resample.append(label)
		resample.append(classNum)
		resample.append(species)
		resTmp = pd.DataFrame(resample)
		pieces.append(resTmp.T)

outputs = pd.concat(pieces, ignore_index = True)
outputs.to_csv('../data/one/file.csv', index = False)

# 直接根据波数合并，不合理，弃用
# for file in files:
# 	if os.path.splitext(file)[1] == '.txt' and specify_str in file:
# 		print ("------Load " + file + "------")
# 		dataPath = filepath + file
# 		print file.split('__')[0]
# 		txt = np.loadtxt(dataPath, delimiter = ', ')
# 		# print txt
# 		txtDF = pd.DataFrame(txt, columns = list('AB')) # 用波数代替columns list('AB')
# 		# print txtDF
# 		txtTmp = pd.DataFrame(txt, index = txtDF['A'])
# 		# print txtTmp
# 		txtDF1 = pd.DataFrame(txtTmp.iloc[:,1])
# 		# print txtDF1
# 		pieces.append(txtDF1.T)
# 		# c = np.vstack((a,b)) # 纵向合并
# outputs = pd.concat(pieces, ignore_index = True)
# # print txtDF

# # print outputs
# outputs.to_csv('../data/one/file.csv', index = False)

# resample test
# x = txtDF['A'].T
# y = txtDF['B'].T
# xmax = math.floor(max(x))
# xmin = math.ceil(min(x))
# xnum = xmax - xmin + 1
# xnew = np.linspace(xmin, xmax, xnum)

# plt.figure()
# plt.plot(x,y,color='red')

# f = interpolate.interp1d(x,y,kind = 'cubic')
# ynew = f(xnew)

# xres = np.linspace(85, 1800, 1716)
# xmin = int(xmin)
# xmax = int(xmax)
# res1 = list(np.zeros(xmin-85))
# res2 = list(np.zeros(1800-xmax))
# resample = res1 + list(ynew) + res2

# plt.plot(xres,resample,'k*')
# plt.show()
