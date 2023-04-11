import csv
csv.field_size_limit(500 * 1024 * 1024)
# 读的字段太大，https://blog.csdn.net/dm_learner/article/details/79028357
# import sys   # 或者
# import csv
# csv.field_size_limit(sys.maxsize)
import numpy as np


# 定义函数
def ReadMyCsv1(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        for i in range(len(row)):       # 转换数据类型
            row[i] = float(row[i])
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


'''
1，小写、人类、统一标识符、顺序
2，去重
'''

# 读数据
AssociationPair1 = []
ReadMyCsv1(AssociationPair1, "miRNA.csv")

AssociationPair2 = []
ReadMyCsv1(AssociationPair2, "MyMiRBase.csv")


num = 0
counter1 = 0
while counter1 < len(AssociationPair1):
    counter2 = 0
    while counter2 < len(AssociationPair2):
        if AssociationPair1[counter1][0].lower() == AssociationPair2[counter2][0].lower():
            AssociationPair1[counter1].append(AssociationPair2[counter2][1])
            num = num + 1
            break
        counter2 = counter2 + 1
    counter1 = counter1 + 1
    print(counter1)

print('num', num)
StorFile(AssociationPair1, 'AllMiSequence.csv')



