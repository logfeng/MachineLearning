# -*- coding: utf-8 -*-
import os
import csv
import numpy as np  
import pandas as pd 

filepath = '../data/one/'
# filepath = '../data/excellent_unoriented/'
files = os.listdir(filepath)
pieces = []
specify_str = '780'
for file in files:
	if os.path.splitext(file)[1] == '.txt' and specify_str in file:
		print ("------Load " + file + "------")
		dataPath = filepath + file
		txt = np.loadtxt(dataPath, delimiter = ', ')
		txtDF = pd.DataFrame(txt, columns = list('AB')) # 用波数代替columns list('AB')
		txtDF.set_index('A')
		# print txtDF
		txtTmp = pd.DataFrame(txtDF['B'], index = list(txtDF['A']))
		# print txtTmp
		txtDF1 = pd.DataFrame(txtDF.iloc[:,1], index = list(txtDF.iloc[:,0]))
		# print txtDF1.T
		pieces.append(txtDF.T)
		# c = np.vstack((a,b)) # 纵向合并
outputs = pd.concat(pieces, ignore_index = True)
# print outputs
outputs.to_csv('../data/one/file.csv', index = False)

#load data
# file=open("D:\pythonTest\data\one\Abelsonite__R070007__Raman__532__0__unoriented__Raman_Data_Processed__27040.txt")
# lines=file.readlines()
# rows=len(lines)
# print rows

# datamat=np.zeros((rows,2))

# row=0
# for line in lines:
#     line=line.strip(', ').split('\t')
#     datamat[row,:]=line[:]
#     row+=1

# print(datamat)
# print(datamat.shape)
