# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:36:58 2020

@author: LXC
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array

from keras.models import Model
from keras.layers import Dense, Input
import IoTData
from sklearn import metrics
import matplotlib.pyplot as plt

np.random.seed(33)  # random seed，to reproduce results.

ENCODING_DIM_INPUT = 6
ENCODING_DIM_OUTPUT = 2
EPOCHS = 100#100
BATCH_SIZE = 50#256

Encoder1 = 13
# Encoder2 = 20
training_precentage=0.90

def train(x_train):
    """
    build autoencoder.
    :param x_train:  the train data
    :return: encoder and decoder
    """
    # input placeholder
    input_image = Input(shape=(ENCODING_DIM_INPUT,))


    # encoding layer
    hidden_layer = Dense(ENCODING_DIM_INPUT, activation='relu')(input_image)
    hidden_layer = Dense(32, activation='relu')(hidden_layer)
    hidden_layer = Dense(16, activation='relu')(hidden_layer)
    hidden_layer = Dense(8, activation='relu')(hidden_layer)
    hidden_layer = Dense(4, activation='relu')(hidden_layer)
    encoder_output = Dense(2)(hidden_layer)
    # encoder_output = Dense(16, activation='relu')(hidden_layer)
    # encoder_output = Dense(ENCODING_DIM_OUTPUT)(hidden_layer)

    # decoding layer
    decoded = Dense(2, activation='relu')(encoder_output)
    decoded = Dense(4, activation='relu')(decoded)
    decoded = Dense(8, activation='relu')(decoded)
    decoded = Dense(16, activation='relu')(decoded)
    decoded = Dense(32, activation='relu')(decoded)
    decode_output = Dense(ENCODING_DIM_INPUT, activation='tanh')(decoded)
    # decode_output = Dense(ENCODING_DIM_INPUT, activation='relu')(hidden_layer)

    # build autoencoder, encoder, decoder
    autoencoder = Model(inputs=input_image, outputs=decode_output)
    encoder = Model(inputs=input_image, outputs=encoder_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    history = autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
    plt.plot(history.history['loss'],
             'b',
             label='Training loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss, [mse]')
    # plt.ylim([0, 0.5])
    plt.savefig('loss.jpg')

    return encoder, autoencoder


#选择最优的epsilon，即：使F1Score最大
def selectThreshold(RecError, ytrue, coefficient, higher_q, IQR):
     '''初始化所需变量'''

     bestEpsilon = 0.
     bestF1 = 0.
     F1 = 0.
     step = (np.max(coefficient)-np.min(coefficient))/1000
     '''计算'''
     for epsilon in np.arange(np.min(coefficient),np.max(coefficient),step):
         threshod = higher_q + epsilon * IQR
         cvPrecision = RecError < threshod
         tp = np.sum((cvPrecision == 1) & (ytrue == 1).ravel()).astype(float)  # sum求和是int型的，需要转为float
         fp = np.sum((cvPrecision == 1) & (ytrue == 0).ravel()).astype(float)
         fn = np.sum((cvPrecision == 0) & (ytrue == 1).ravel()).astype(float)
         precision = tp/(tp+fp)  # 精准度
         recision = tp/(tp+fn)   # 召回率
         F1 = (2*precision*recision)/(precision+recision)  # F1Score计算公式
         print('F1:', F1)
         if F1 > bestF1:  # 修改最优的F1 Score
             bestF1 = F1
             bestEpsilon = epsilon
             bestEpsilon = epsilon
             bestthreshod = threshod
     return bestEpsilon,bestF1, bestthreshod





if __name__ == '__main__':
###############
    path1 = 'D:\处理后数据/18-10-19_ok.csv'
    path2 = 'D:\处理后数据/18-10-24_ok.csv'
    dftrain = pd.read_csv(path1, header=None)  # 全部是正常数据
    dftest = pd.read_csv(path2, header=None)  # 混合异常数据

    df=pd.merge(dftrain,dftest, how = 'outer')
    df_train = df.sample(frac=0.8, random_state=42)
    df_test = df.loc[~df.index.isin(df_train.index)]
    df_valid = df_train.sample(frac=0.2, random_state=42)
#############

    trainx, trainy = IoTData.get_train()
    testx, testy = IoTData.get_test()
    validx, validy = IoTData.get_valid()


    # Step4： train
    encoder, autoencoder = train(trainx)
    # vailda_pred = autoencoder.predict(vaildx)
    # test and plot
    X_pred = autoencoder.predict(validx)
    X_true = pd.DataFrame(validx)
    scored = pd.DataFrame(index=X_true.index)
    scored['Loss_mae'] = np.linalg.norm((X_pred - validx), axis=1)
    ThsholdNormal = scored['Loss_mae']
    scoredmean = np.mean(ThsholdNormal)
    scoredvar = np.var(ThsholdNormal)

#################################
     #首先利用验证集生成正常数据预测的重构误差，依次进行阈值的计算
    lower_q = np.quantile(ThsholdNormal, 0.25, interpolation='lower')  # 下四分位数
    higher_q = np.quantile(ThsholdNormal, 0.75, interpolation='higher')  # 上四分位数
    IQR = higher_q - lower_q
    print("lower_q, higher_q, IQR: ", lower_q, higher_q, IQR)
##############################3

    #计算阈值，并且用测试集测试结果
    X_pred_test = autoencoder.predict(testx)
    X_true_test = pd.DataFrame(testx)
    scored = pd.DataFrame(index=X_true_test.index)
    scored['Loss_mae'] = np.linalg.norm((X_pred_test - testx), axis=1)


###########################################
    ThsholdAttack = scored['Loss_mae']
    coefficient = [0, 1, 2, 3, 4, 5]
    bestEpsilon,bestF1, bestthreshod = selectThreshold(ThsholdAttack, testy, coefficient, higher_q, IQR)
    print('bestthreshod:', bestthreshod)
    print('bestEpsilon:', bestEpsilon)
    print('bestF1:', bestF1)
    #scored['Threshold'] = bestthreshod

############################################
    #scored['Threshold'] =0.1
    scored['Threshold'] = np.percentile(ThsholdNormal, 90)
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']

    label_predict = []
    scored.head()
    ReconstructionError = scored['Anomaly']
    for i in range(len(ReconstructionError)):
        if (ReconstructionError[i] == True):
            label_predict.append(1)
        else:
            label_predict.append(0)




    label_true = testy

    accuracy = []
    precision = []
    recall = []
    f1 = []

    accuracy.append(metrics.accuracy_score(label_true, label_predict))
    precision.append(metrics.precision_score(label_true, label_predict))
    recall.append(metrics.recall_score(label_true, label_predict))
    f1.append(metrics.f1_score(label_true, label_predict, labels=[0, 1]))



    print("**********************************************************************")
    print("Accuracy on Testing Data :", accuracy)
    print("precision : ", precision)
    print("recall: ",recall)
    print("f1: ", f1)




