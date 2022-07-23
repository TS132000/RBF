# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import keras
from keras import layers
from keras.layers import Activation, Dense
from keras.optimizers import RMSprop
import rbflayer, kmeans_initializer

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from scipy import * 
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt  
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICE"]  = '0'



# load feature data
Feature_A = pd.read_excel(io=r'Feature_A.xlsx')
Feature_B = pd.read_excel(io=r'Feature_B.xlsx')
Feature_C = pd.read_excel(io=r'Feature_c.xlsx')

Feature_A = Feature_A.rename(columns={"Isa_A": "Isa", "Isd_A": "Isd","Iar_A": "Iar","Pvd_A": "Pvd"})
Feature_B = Feature_B.rename(columns={"Isa_B": "Isa", "Isd_B": "Isd","Iar_B": "Iar","Pvd_B": "Pvd"})
Feature_C = Feature_C.rename(columns={"Isa_C": "Isa", "Isd_C": "Isd","Iar_C": "Iar","Pvd_C": "Pvd"})

Feature = pd.concat([
    Feature_A,
    Feature_B,
    Feature_C,
]) 

label_y = Feature['label']
X_feat =  Feature[['负载率','Isa','三相电流不平衡度','Isd','Iar','Pvd','负载率差值']]
# float64->float32
#X_feat[X_feat.select_dtypes(np.float64).columns] = X_feat.select_dtypes(np.float64).astype(np.float32)
# inf,nan数据填充
X_feat = (X_feat.replace([np.inf, -np.inf], np.nan)).fillna(value = 0) 

# SMOTE样本均衡化
from collections import Counter
# 查看所生成的样本类别分布，属于类别不平衡数据
print(Counter(label_y))


# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_resample(X_feat, label_y)


# RBF
import rbflayer, kmeans_initializer

x_iris = X_smo
y_iris = y_smo

# train
kfold = KFold(n_splits=2)
history_rbf = []

for train_index, test_index in kfold.split(x_iris):
    X_train,X_test = x_iris.iloc[train_index], x_iris.iloc[test_index]
    Y_train,Y_test = y_iris.iloc[train_index], y_iris.iloc[test_index]
    # creating RBF network
    RBF = rbflayer.RBFLayer(10,
                      initializer=kmeans_initializer.InitCentersKMeans(x_iris),
                      betas=2.0,
                      input_shape=([7]))

    model = keras.models.Sequential()
    model.add(RBF)
    model.add(layers.Dense(1,activation = 'softmax'))
    model.compile(optimizer=RMSprop(),loss ='binary_crossentropy', metrics=['acc'])

    history = model.fit(X_train,Y_train,
                       epochs = 50,
                       batch_size = 8,
                       validation_data = (X_test,Y_test)
                       )
    history_rbf.append(history)

pd.DataFrame(history_rbf[0].history).plot()
model.save("RBF.h5")