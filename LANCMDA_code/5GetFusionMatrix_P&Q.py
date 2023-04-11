import numpy as np
import csv
# csv.field_size_limit(500 * 1024 * 1024)
# 读的字段太大，https://blog.csdn.net/dm_learner/article/details/79028357
import os
import time
import numpy as np

import math
import random
def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = float(row[counter])      # 转换数据类型
            counter = counter + 1
        SaveList.append(row)
    return

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
A=[]
R=[]
alpha=0.95
ReadMyCsv2(A,"A.csv")
ReadMyCsv2(R,"R.csv")
A=np.array(A)
R=np.array(R)
P1=alpha*A+(1-alpha)*R
StorFile(P1,"P1.csv")

P2=np.dot(alpha*P1,A)+(1-alpha)*R
StorFile(P2,"P2.csv")

P3=np.dot(alpha*P2,A)+(1-alpha)*R
StorFile(P3,"P3.csv")

P4=np.dot(alpha*P3,A)+(1-alpha)*R
StorFile(P4,"P4.csv")

P5=np.dot(alpha*P4,A)+(1-alpha)*R
StorFile(P5,"P5.csv")

# P6=np.dot(alpha*P5,A)+(1-alpha)*R
# StorFile(P6,"P6.csv")
#
# P7=np.dot(alpha*P6,A)+(1-alpha)*R
# StorFile(P7,"P7.csv")

beta=0.94
Q=(beta**(1))*P1+(beta**(2))*P2+(beta**(3))*P3+(beta**(4))*P4+(beta**(5))*P5

StorFile(Q,"Q.csv")
