import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Column names of the dataframe
def _col_names():
    """Column names of the dataframe"""
    return ["1", "2", "3", "4", "5",
            "6", "label"]

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
    fea = df[result]
    fea = (fea).iloc[:,:].values
    dummies = df[target]
    dummies = (dummies).iloc[:].values
    return fea, dummies

def _get_dataset(scale):
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
    df_benign = pd.read_csv("D:\处理后数据/18-10-19_ok.csv", header=None, names=col_names)
    df_attack = pd.read_csv("D:\处理后数据/18-10-24_ok.csv", header=None, names=col_names)
    labels = np.zeros(df_benign.shape[0])
    df_benign['label'] = labels
    df = pd.merge(df_benign, df_attack, how = 'outer')
    print(df.shape[0])
    df_train = df.sample(frac=0.9, random_state=42)
    print(df_train.shape[0])
    df_test = df.loc[~df.index.isin(df_train.index)]
    print(df_test.shape[0])
    df_valid = df_train.sample(frac=0.1, random_state=42)

    x_train, y_train = _to_xy(df_train, target='label')
    x_valid, y_valid = _to_xy(df_valid, target='label')
    x_test, y_test = _to_xy(df_test, target='label')

    y_train = y_train.flatten().astype(int)
    y_valid = y_valid.flatten().astype(int)
    y_test = y_test.flatten().astype(int)
    x_train = x_train[y_train != 1]
    y_train = y_train[y_train != 1]
    x_valid = x_valid[y_valid != 1]
    y_valid = y_valid[y_valid != 1]

    if scale:
        print("Scaling KDD dataset")
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_valid'] = x_valid.astype(np.float32)
    dataset['y_valid'] = y_valid.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)


    return dataset



def _adapt_ratio(x, y, rho=0.2):
    """Adapt the ratio of normal/anomalous data"""

    # Normal data: label =0, anomalous data: label =1

    rng = np.random.RandomState(42) # seed shuffling

    inliersx = x[y == 0]
    inliersy = y[y == 0]
    outliersx = x[y == 1]
    outliersy = y[y == 1]

    size_outliers = outliersx.shape[0]
    inds = rng.permutation(size_outliers)
    outliersx, outliersy = outliersx[inds], outliersy[inds]

    size_x = inliersx.shape[0]
    out_size_x = int(size_x*rho/(1-rho))

    out_sample_x = outliersx[:out_size_x]
    out_sample_y = outliersy[:out_size_x]

    x_adapted = np.concatenate((inliersx,out_sample_x), axis=0)
    y_adapted = np.concatenate((inliersy,out_sample_y), axis=0)

    size_x = x_adapted.shape[0]
    inds = rng.permutation(size_x)
    x_adapted, y_adapted = x_adapted[inds], y_adapted[inds]

    return x_adapted, y_adapted

def _get_adapted_dataset(split, scale):
    """ Gets the adapted dataset for the experiments

    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    # print("_get_adapted",scale)
    dataset = _get_dataset(scale)

    key_img = 'x_' + split
    key_lbl = 'y_' + split
    if split == 'test':
        dataset[key_img], dataset[key_lbl] = _adapt_ratio(dataset[key_img],
                                                    dataset[key_lbl])
    return (dataset[key_img], dataset[key_lbl])

def get_train(label=0, scale=True, *args):
    """Get training dataset for KDD 10 percent"""
    return _get_adapted_dataset("train", scale)

def get_test(label=0, scale=True, *args):
    """Get testing dataset for KDD 10 percent"""
    return _get_adapted_dataset("test", scale)

def get_valid(label=0, scale=True, *args):
    """Get validation dataset for KDD 10 percent"""
    return _get_adapted_dataset("valid", scale)

def get_shape_input():
    """Get shape of the dataset for KDD 10 percent"""
    return (None, 70)

def get_shape_label():
    """Get shape of the labels in KDD 10 percent"""
    return (None,)

def get_anomalous_proportion():
    return 0.2

 
