import logging
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from scipy import stats
import IoTData

def get_train(*args):
    """Get training dataset for KDD 10 percent"""
    return _get_adapted_dataset("train")

def get_test(*args):
    """Get testing dataset for KDD 10 percent"""
    return _get_adapted_dataset("test")

def get_shape_input():
    """Get shape of the dataset for KDD 10 percent"""
    return (None, 121)

def get_shape_label():
    """Get shape of the labels in KDD 10 percent"""
    return (None,)

def _get_dataset():
    """ Gets the basic dataset
    Returns :
            dataset (dict): containing the data
                dataset['x_train'] (np.array): training images shape
                (?, 120)
                dataset['y_train'] (np.array): training labels shape
                (?,)
                dataset['x_test'] (np.array): testing images shape
                (?, 120)
                dataset['y_test'] (np.array): testing labels shape
                (?,)
    """
    col_names = _col_names()
    df = pd.read_csv("D:/处理后数据/18-10-19_ok.csv", header=None, names=col_names)
    
    text_l = ["1", "2", "3", "4", "5",
            "6"]

    for name in text_l:
        _encode_text_dummy(df, name)
    
    labels = df['label'].copy()
    # Given the ratio of normal(less) and abnormal samples(more), suppose preponderant samples are normal
    # Isolation Tree regards the dominant labels as 1(normal), and the less as -1(abnormal)
    labels[labels != 'normal.'] = 1
    labels[labels == 'normal.'] = -1

    df['label'] = labels

    df_train = df.sample(frac=0.5, random_state=42)
    df_test = df.loc[~df.index.isin(df_train.index)]
    
    x_train, y_train = _to_xy(df_train, target='label')
    y_train = y_train.flatten().astype(int)
    x_test, y_test = _to_xy(df_test, target='label')
    y_test = y_test.flatten().astype(int)

    # x_train = x_train[y_train != 1]
    # y_train = y_train[y_train != 1]
    
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    scaler.transform(x_train)
    scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    return dataset

def _get_adapted_dataset(split):
    """ Gets the adapted dataset for the experiments

    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    dataset = IoTData._get_dataset()#_get_dataset()
    key_img = 'x_' + split
    key_lbl = 'y_' + split

    if split != 'train':
        dataset[key_img], dataset[key_lbl] = IoTData._adapt_ratio(dataset[key_img],
                                                    dataset[key_lbl])

    return (dataset[key_img], dataset[key_lbl])

def _encode_text_dummy(df, name):
    """Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1]
    for red,green,blue)
    """
    dummies = pd.get_dummies(df.loc[:,name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)

def _col_names():
    """Column names of the dataframe 42-dim"""
    return ["1", "2", "3", "4", "5",
            "6", "label"]

def _adapt(x, y, rho=0.1):
    """Adapt the ratio of normal/anomalous data"""

    # Normal data: label =0, anomalous data: label =1

    rng = np.random.RandomState(42) # seed shuffling

    inliersx = x[y == 1]
    inliersy = y[y == 1]
    outliersx = x[y == -1]
    outliersy = y[y == -1]

    size_outliers = outliersx.shape[0]
    inds = rng.permutation(size_outliers)
    outliersx, outliersy = outliersx[inds], outliersy[inds]

    size_test = inliersx.shape[0]
    out_size_test = int(size_test*rho/(1-rho))
    outestx = outliersx[:out_size_test]
    outesty = outliersy[:out_size_test]

    testx = np.concatenate((inliersx,outestx), axis=0)
    testy = np.concatenate((inliersy,outesty), axis=0)

    size_test = testx.shape[0]
    inds = rng.permutation(size_test)
    testx, testy = testx[inds], testy[inds]

    return testx, testy

trainx, trainy = IoTData.get_train()#get_train()
testx, testy = IoTData.get_test()#get_test()
clf = svm.OneClassSVM(nu=0.95 * 0.1 + 0.05,kernel="rbf", gamma=0.1)
clf.fit(trainx)
y_pred = clf.predict(testx)
'''
with open("pred.txt", 'w') as f:
    for i in y_pred:
        f.write(str(i) + '\n')
with open("testy.txt", 'w') as f:
    for i in testy:
        f.write(str(i) + '\n')
'''
print(testy.shape,y_pred.shape)
print('testy 1:',(testy == 1).sum())#测试集的异常 1为异常0为正常
print('y_pred 1:',(y_pred == -1).sum())#预测的异常 -1为异常1为正常
posi_num = 0

for i in range(0, len(testy)):
    if testy[i] == -y_pred[i] and testy[i] == 1:#异常值 预测正确的时候
        posi_num += 1
print('posi_num', posi_num)
n_errors = (y_pred-testy!=1).sum()-posi_num#异常值预测错误

print("Total errors:", n_errors, "Accuracy:", 1 - n_errors / testx.shape[0], 'Precision:', posi_num / (y_pred == -1).sum())

n_errors = (y_pred != testy).sum()

precision=posi_num / (y_pred == -1).sum()
recall=posi_num / (testy == 1).sum()
print("Total errors:", n_errors, "Accuracy:", 1 - n_errors / testx.shape[0], 'Precision:', posi_num / (y_pred == -1).sum(),'Recall', posi_num / (testy == 1).sum(),
      "F1-Score",2*precision*recall/(precision+recall))
