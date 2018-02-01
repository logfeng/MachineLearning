# -*- coding: utf-8 -*-
import os
import csv
import numpy as np  
import pandas as pd 

filepath = 'D:/pythonTest/data/one/'
files = os.listdir(filepath)
pieces = []
for file in files:
	if os.path.splitext(file)[1] == '.txt':
		print ("------Load " + file + "------")
		dataPath = filepath + file
		txt = np.loadtxt(dataPath, delimiter=', ')
		txtDF = pd.DataFrame(txt)
		pieces.append(txtDF)
		print txtDF.T  # 转置
		output = txtDF.T
		# c = np.vstack((a,b)) # 纵向合并
        # output.to_csv('D:/pythonTest/data/one/file.csv', index=False)
outputs = pd.concat(pieces, ignore_index = True)
print outputs
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
