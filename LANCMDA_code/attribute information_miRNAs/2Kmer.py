import csv
csv.field_size_limit(500 * 1024 * 1024)
# 读的字段太大，https://blog.csdn.net/dm_learner/article/details/79028357
import os
import time
import numpy as np
# import pandas as pd
import math
import random
from MyKmer import MyKmer

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


# 读取源文件
BiomoleculesSequence = []
ReadMyCsv(BiomoleculesSequence, "AllMiSequence.csv")
print('len(BiomoleculesSequence)', len(BiomoleculesSequence))
print(BiomoleculesSequence[0])

BiomoleculesKmer = []
counter = 0
while counter < len(BiomoleculesSequence):
    try:        # 有序列则kmer
        sequence = MyKmer(BiomoleculesSequence[counter][1])
        pair = []
        pair.append(BiomoleculesSequence[counter][0])
        pair.extend(sequence)
        BiomoleculesKmer.append(pair)
    except:     # 没有不加
        pair = []
        pair.append(BiomoleculesSequence[counter][0])
        pair.extend(np.zeros((1, 64), dtype=int)[0])            # 没有的填0
        BiomoleculesKmer.append(pair)
    print(counter)
    counter = counter + 1

print(len(BiomoleculesKmer))
print(len(BiomoleculesKmer[0]))
StorFile(BiomoleculesKmer, 'AllMiKmer.csv')