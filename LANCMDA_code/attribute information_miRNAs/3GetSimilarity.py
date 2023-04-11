import numpy as np
import csv
# csv.field_size_limit(500 * 1024 * 1024)
# 读的字段太大，https://blog.csdn.net/dm_learner/article/details/79028357
import os
import time
import numpy as np

import math
import random
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
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
ReadMyCsv(AllMiKmer,"AllMiKmer.csv")
AllMiKmer_Jac=np.zeros((901,901),dtype=float)

for i in range(901):
    print(i)
    for j in range(901):
        sim=np.vstack([np.array(AllMiKmer[i][1:]),np.array(AllMiKmer[j][1:])])
        AllMiKmer_Jac[i][j] = pdist(sim, 'euclidean')

AllMiKmer_Jac=AllMiKmer_Jac.tolist()
StorFile(AllMiKmer_Jac,"AllMiKmer_Euc.csv")
