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
S=[]
Structure_transition_matrix=[]
ReadMyCsv(S, "S.csv")
#根据公式一计算
A=[]
for i in range(901+877):
	pair=[]
	for j in range(901+877):
		pair.append(0)
	A.append(pair)

for i in range(901+877):
    sum = 0
    for j in range(901+877):
        sum=float(S[i][j])+sum

    for m in range(901+877):
        print(i,sum)
        if sum==0:#分母有0的情况
            A[i][m]=0
        else:
            A[i][m]=float(S[i][m])/sum

StorFile(A,"A.csv")
