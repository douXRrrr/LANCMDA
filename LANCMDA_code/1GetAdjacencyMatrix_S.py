import numpy as np
import csv
# csv.field_size_limit(500 * 1024 * 1024)
# 读的字段太大，https://blog.csdn.net/dm_learner/article/details/79028357
import os
import time
import numpy as np

import math
import random


# 定义函数
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

AM=[]   #Association Matrix
ReadMyCsv(AM, "Association Matrix.csv")
AM_T=[]  #关联矩阵转置
AM_T=list(map(list, zip(*AM)))[::-1]
#Association Matrix的行数为901，列数为877，我们需要拼接成一个（1057+850,1057+850）的结构相似矩阵，缺失处补0

ZeroMatrix1057_1057=[]#生成901*901和877*877的0矩阵
ZeroMatrix850_850=[]
for i in range(901):
	pair=[]
	for j in range(901):
		pair.append(0)
	ZeroMatrix1057_1057.append(pair)
for i in range(877):
	pair=[]
	for j in range(877):
		pair.append(0)
	ZeroMatrix850_850.append(pair)
structure_adjacency_matrix=[]
for i in range(901):#拼接上半矩阵
    pair=[]
    pair.extend(ZeroMatrix1057_1057[i][:]+AM[i][:])

    structure_adjacency_matrix.append(pair)
for i in range(877):#拼接下半矩阵
    pair=[]
    pair.extend(AM_T[i][:]+ZeroMatrix850_850[i][:])

    structure_adjacency_matrix.append(pair)
StorFile(structure_adjacency_matrix,"S.csv")
