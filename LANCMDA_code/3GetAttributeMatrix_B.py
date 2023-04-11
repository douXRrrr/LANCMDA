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

AllMiKmer=[]
ReadMyCsv(AllMiKmer, "AllMiKmer_Euc.csv")
AllDiseaseMesh=[]
ReadMyCsv(AllDiseaseMesh, "DiseaseSimilarityModel2.csv")


ZeroMatrix1057_850=[]
ZeroMatrix850_1057=[]
for i in range(901):
	pair=[]
	for j in range(877):
		pair.append(0)
	ZeroMatrix1057_850.append(pair)
for i in range(877):
	pair=[]
	for j in range(901):
		pair.append(0)
	ZeroMatrix850_1057.append(pair)
structure_adjacency_matrix=[]
for i in range(901):#拼接上半矩阵
    pair=[]
    pair.extend(AllMiKmer[i][:]+ZeroMatrix1057_850[i][:])

    structure_adjacency_matrix.append(pair)
for i in range(877):#拼接下半矩阵
    pair=[]
    pair.extend(ZeroMatrix850_1057[i][:]+AllDiseaseMesh[i][1:])

    structure_adjacency_matrix.append(pair)
StorFile(structure_adjacency_matrix,"B.csv")
