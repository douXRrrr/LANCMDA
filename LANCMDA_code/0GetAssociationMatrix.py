import numpy as np
import csv
import sys
csv.field_size_limit(500 * 1024 * 1024)
# 读的字段太大，https://blog.csdn.net/dm_learner/article/details/79028357
import os
import time
import numpy as np
# csv.field_size_limit(sys.maxsize)
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

miRNA = []
disease=[]
HMDD=[]
ReadMyCsv(miRNA, "miRNA.csv")
ReadMyCsv(disease, "disease.csv")
ReadMyCsv(HMDD, "MDCuiMiDisease.csv")
miRNA_num=len(miRNA)
disease_num=len(disease)
association=np.zeros((miRNA_num,disease_num))#miRNA个数901，disease个数877，初始化为0矩阵

for i in range(miRNA_num):#行循环
	print(i)
	for j in range(disease_num):#列循环
		for p in range(len(HMDD)):#如果miRNA和disease对存在在HMDD中则赋值为1
			if str(miRNA[i][0])==HMDD[p][0] and str(disease[j][0])==HMDD[p][1] :
				association[i][j] = '1'
				break
print(association)
association=association.tolist()
StorFile(association,'Association Matrix.csv') # Association Matrix
