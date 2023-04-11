import os
import time
import numpy as np
import pandas as pd
import csv
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


# 读取源文件
FinalAllDisease = []
ReadMyCsv(FinalAllDisease, "disease.csv")
FinalAllDisease = np.array(FinalAllDisease)[:, 0]
print(len(FinalAllDisease))
print(FinalAllDisease[1])
# FinalAllDisease = FinalAllDisease[0:100]


DiseaseMeSHTreeStructure = []
ReadMyCsv(DiseaseMeSHTreeStructure, "MeSHTreeStructureLow.csv")
print(len(DiseaseMeSHTreeStructure))
print(DiseaseMeSHTreeStructure[0])

AllDisease = FinalAllDisease
mesh = DiseaseMeSHTreeStructure

# 构建disease idGroup，有对应id的生成id group，没有的加入[0]
print('构建disease idGroup，有对应id的生成id group，没有的加入[0]')
DiseaseAndMeshID = []
counter1 = 0
while counter1 < len(AllDisease):
    DiseaseAndMeshPair = []
    DiseaseAndMeshID.append(DiseaseAndMeshPair)
    DiseaseAndMeshID[counter1].append(AllDisease[counter1])
    counter2 = 0
    flag = 0
    while counter2 < len(mesh):#遍历整个mesh，寻找相同疾病的所有id
        if (mesh[counter2][0] == DiseaseAndMeshID[counter1][0]) & (flag == 1):#加入
            DiseaseAndMeshID[counter1][1].append(mesh[counter2][1])
        if (mesh[counter2][0] == DiseaseAndMeshID[counter1][0]) & (flag == 0):#新建mesh id 列表
            MeshID = []
            MeshID.append(mesh[counter2][1])
            DiseaseAndMeshID[counter1].append(MeshID)
            flag = 1
        if (counter2 == len(mesh) - 1) & (len(DiseaseAndMeshID[counter1]) == 1):    # 当遍历到最后一个长度仍为1，增加一个0
            DiseaseAndMeshID[counter1].append(0)
        counter2 = counter2 + 1
    print(counter1)
    counter1 = counter1 + 1
print('DiseaseAndMeshID')
print(len(DiseaseAndMeshID))
StorFile(DiseaseAndMeshID, 'DiseaseAndMeshID2.csv')

# 构建疾病的DAGs
# 构建dags的根节点
print('构建DAG的根节点')
DAGs = []
counter1 = 0
while counter1 < len(AllDisease):
    group = []
    group.extend(DiseaseAndMeshID[counter1])
    group.append(0)
    group1 = []
    group1.append(group)
    DAGs.append(group1)
    print(counter1)
    counter1 = counter1 + 1
print('len(DAGs)的叶子', len(DAGs))
StorFile(DAGs, 'DAGsLeaf2.csv')



print('构建DAG')
counter = 0
while counter < len(DAGs):
    if DAGs[counter][0][1] == 0:
        counter = counter + 1
        continue
    counter1 = 0
    while counter1 < len(DAGs[counter]):  #################
        counter2 = 0
        while counter2 < len(DAGs[counter][counter1][1]):  ###################只对一个节点扩展只能生成的二层信息
            layer = DAGs[counter][counter1][2]  #######################
            # if len(DAGs[0][counter1][1][counter2]) <= 3:
            #     break
            if len(DAGs[counter][counter1][1][counter2]) > 3:  ####################
                NID = DAGs[counter][counter1][1][counter2]  #####################
                L = len(NID)
                NID = NID[0:L - 4]  # 把id减3
                counter3 = 0
                flag = 1  # 默认不在
                while counter3 < len(mesh):  # 判断nid是否在mesh中，如果在求出疾病名，如果不在，跳出循还
                    if NID == mesh[counter3][1]:
                        flag = 0  # 由counter3找对应的疾病名
                        num = counter3
                        DiseaseName = mesh[counter3][0]
                        break
                    counter3 = counter3 + 1

                flag2 = 0  # 默认在dags不存在
                counter5 = 0
                while counter5 < len(DAGs[counter]):  # 找到对应疾病的名字后查找dags看是否已经出现，出现了就不加了
                    if DAGs[counter][counter5][0] == DiseaseName:  #########################
                        flag2 = 1  # dags中出现了
                        break
                    counter5 = counter5 + 1

                if flag == 0:
                    if flag2 == 0:
                        counter6 = 0    # 遍历mesh，寻找disease对应的id
                        IDGroup = []
                        while counter6 < len(mesh):
                            if DiseaseName == mesh[counter6][0]:
                                IDGroup.append(mesh[counter6][1])
                            counter6 = counter6 + 1
                        DiseasePoint = []
                        layer = layer + 1
                        DiseasePoint.append(DiseaseName)
                        DiseasePoint.append(IDGroup)
                        DiseasePoint.append(layer)
                        DAGs[counter].append(DiseasePoint)  ######################

            counter2 = counter2 + 1
        counter1 = counter1 + 1
    print(counter)
    counter = counter + 1
print('DAGs', len(DAGs))
StorFile(DAGs, 'DAGs2.csv')

# 构建model1
# 构建DV(disease value)，通过AllDisease构建的DiseaseAndMesh和DAGs，所以疾病顺序都一样，通过dags的layer构建DiseaseValue
DiseaseValue = []
counter = 0
while counter < len(AllDisease):
    if DAGs[counter][0][1] == 0:
        DiseaseValuePair = []
        DiseaseValuePair.append(AllDisease[counter])
        DiseaseValuePair.append(0)
        DiseaseValue.append(DiseaseValuePair)
        counter = counter + 1
        continue
    counter1 = 0
    DV = 0
    while counter1 < len(DAGs[counter]):
        DV = DV + math.pow(0.5, DAGs[counter][counter1][2])
        counter1 = counter1 + 1
    DiseaseValuePair = []
    DiseaseValuePair.append(AllDisease[counter])
    DiseaseValuePair.append(DV)
    DiseaseValue.append(DiseaseValuePair)
    print(counter)
    counter = counter + 1
print('len(DiseaseValue)', len(DiseaseValue))
StorFile(DiseaseValue, 'DiseaseValue2.csv')


# 生成两个疾病DAGs相同部分的DV
print('生成两个疾病DAGs相同部分的DV')
SameValue1 = []
counter = 0
while counter < len(AllDisease):
    RowValue = []
    if DiseaseValue[counter][1] == 0:           # 没有mesh id，整行都为0
        counter1 = 0
        while counter1 < len(AllDisease):
            RowValue.append(0)
            counter1 = counter1 + 1
        SameValue1.append(RowValue)
        counter = counter + 1
        continue
    counter1 = 0
    while counter1 < len(AllDisease):#疾病counter和疾病counter1之间的共同节点
        if DiseaseValue[counter1][1] == 0:  # 没有mesh id，此点为0
            RowValue.append(0)
            counter1 = counter1 + 1
            continue
        DiseaseAndDiseaseSimilarityValue = 0
        counter2 = 0
        while counter2 < len(DAGs[counter]):#疾病counter的所有DAGs的节点
            counter3 = 0
            while counter3 < len(DAGs[counter1]):#疾病counter1的所有DAGs的节点
                if DAGs[counter][counter2][0] == DAGs[counter1][counter3][0]:#找出共同节点
                    DiseaseAndDiseaseSimilarityValue = DiseaseAndDiseaseSimilarityValue + math.pow(0.5, DAGs[counter][counter2][2]) + math.pow(0.5, DAGs[counter1][counter3][2]) #自己和自己的全部节点相同，对角线即DiseaseValue的两倍
                counter3 = counter3 + 1
            counter2 = counter2 + 1
        RowValue.append(DiseaseAndDiseaseSimilarityValue)
        counter1 = counter1 + 1
    SameValue1.append(RowValue)
    print(counter)
    counter = counter + 1
print('SameValue1')
StorFile(SameValue1, 'Samevalue12.csv')


# 生成model1
print('生成model1')
DiseaseSimilarityModel1 = []
counter = 0
while counter < len(AllDisease):
    RowValue = []
    if DiseaseValue[counter][1] == 0:           # 没有mesh id，整行都为0
        counter1 = 0
        while counter1 < len(AllDisease):
            RowValue.append(0)
            counter1 = counter1 + 1
        DiseaseSimilarityModel1.append(RowValue)
        counter = counter + 1
        continue
    counter1 = 0
    while counter1 < len(AllDisease):
        if DiseaseValue[counter1][1] == 0:  # 没有mesh id，此点为0
            RowValue.append(0)
            counter1 = counter1 + 1
            continue
        value = SameValue1[counter][counter1] / (DiseaseValue[counter][1] + DiseaseValue[counter1][1])
        RowValue.append(value)
        counter1 = counter1 + 1
    DiseaseSimilarityModel1.append(RowValue)
    print(counter)
    counter = counter + 1
print('DiseaseSimilarityModel1，行数', len(DiseaseSimilarityModel1))
print('DiseaseSimilarityModel1[0]，列数', len(DiseaseSimilarityModel1[0]))


counter = 0
while counter < len(DiseaseSimilarityModel1):
    Row = []
    Row.append(AllDisease[counter])
    Row.extend(DiseaseSimilarityModel1[counter])
    DiseaseSimilarityModel1[counter] = Row
    counter = counter + 1


StorFile(DiseaseSimilarityModel1, 'DiseaseSimilarityModel2.csv')

