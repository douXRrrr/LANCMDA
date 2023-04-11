import os               # https://blog.csdn.net/marsjhao/article/details/68928486
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # cpu
import numpy as np
# from keras.datasets import mnist
from keras.models import Model  # 泛型模型
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
# import pandas as pd
import csv
import math
import random
from keras import regularizers
np.random.seed(1337)  # for reproducibility

# 定义函数
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return




# 数据预处理
# 读数据
SampleFeature = []
ReadMyCsv(SampleFeature, "Q.csv")
x = np.array(SampleFeature)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, x, test_size=0.2)    # 切分数据集进行训练，用全部数据集x进行“预测”


x_train = x_train.astype('float32') / 1  # minmax_normalized
x_test = x_test.astype('float32') / 1  # minmax_normalized
# x_train = x_train.astype('float32') / 255. - 0.5  # minmax_normalized
# x_test = x_test.astype('float32') / 255. - 0.5  # minmax_normalized

print(x_train.shape)        # (60000, 784)
print(x_test.shape)         # (10000, 784)

# 压缩特征维度至128维
encoding_dim = 300

# this is our input placeholder
input_img = Input(shape=(len(x[0]),))

# 编码层1024  2048
encoded = Dense(1024, activation='relu',activity_regularizer = regularizers.l1(10e-6))(input_img)
encoded = Dense(1024, activation='relu')(encoded)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dense(300, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# 解码层
decoded = Dense(300, activation='relu')(encoder_output)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(1024, activation='relu')(decoded)
decoded = Dense(len(x[0]), activation='tanh')(decoded)               #############

# 构建自编码模型
autoencoder = Model(inputs=input_img, outputs=decoded)

# 构建编码模型
encoder = Model(inputs=input_img, outputs=encoder_output)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(x, x, epochs=100, batch_size=128, shuffle=True, validation_data=(x_test, x_test))

# plotting
encoded_imgs = encoder.predict(x)

NewEncoded_imgs = []
counter = 0
while counter < len(encoded_imgs):
    Row = []
    # Row.append(SampleFeature[counter][0])
    Row.extend(encoded_imgs[counter])
    NewEncoded_imgs.append(Row)
    counter = counter + 1

print('len(NewEncoded_imgs)', len(NewEncoded_imgs))
print('len(NewEncoded_imgs[0)', len(NewEncoded_imgs[0]))

storFile(NewEncoded_imgs, 'Q_AE.csv')