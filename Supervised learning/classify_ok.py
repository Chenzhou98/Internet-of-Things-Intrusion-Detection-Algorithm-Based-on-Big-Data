
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  train_test_split
from sklearn.tree import export_graphviz
import pandas as pd
import graphviz

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from pylab import *
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import time


path1 = 'd:/18-10-24test1.csv'
path2 = 'd:/18-10-21test1.csv'
dftrain = pd.read_csv(path1, header=None)
dftest = pd.read_csv(path2, header=None)
data = pd.merge(dftrain, dftest)

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


#print(dftrain)
#print(dftest)

#dftest.drop(9, inplace=True, axis=1)
#print(dftest)

#dftest.rename(columns = {10:9}, inplace = True)
#print(dftest)

data = pd.merge(dftrain, dftest, how = 'outer')

#print(data.head(10))
data.shape

#dftrain.shape

#dftest.shape

#dftrain.iloc[0,:]

#type(dftrain.iloc[0,3])

#dftrain.shape

#dftrain.iloc[:,0]

data = data.iloc[:,3:] #提取所有行，和第三列以后的数据（去除流序号和IP起始目的地址）

#data.head()

#data.iloc[:,6]

#data.head()

#dftest.drop(9, inplace=True, axis=1)

df_train = data.sample(frac=0.9, random_state=42)
print(df_train.shape[0])
df_train.head
df_test = data.loc[~data.index.isin(df_train.index)]
print(df_test.shape[0])
df_test.head
df_valid = df_train.sample(frac=0.1, random_state=42) #注意invalidation
print(df_valid.shape[0])
df_valid.head


#df_train.head()

#data.drop([0,1,2], inplace=True, axis=1)
#data.head()

df_train.head()

#type(data.iloc[0,0])

x_train, y_train = _to_xy(df_train, target=9)
x_valid, y_valid = _to_xy(df_valid, target=9)
x_test, y_test = _to_xy(df_test, target=9)




def DecisionTree(x_train, y_train, x_test, y_test):

    tree = DecisionTreeClassifier(random_state=0)

    tree.fit(x_train, y_train)
    y_predict=tree.predict(x_test)

    accuracy = []
    precision = []
    recall = []
    f1 = []
    print('label_true', np.sum(y_test))

    print('label_predict', np.sum(y_predict))

    accuracy.append(metrics.accuracy_score(y_test, y_predict))
    precision.append(metrics.precision_score(y_test, y_predict))
    recall.append(metrics.recall_score(y_test, y_predict))
    f1.append(metrics.f1_score(y_test, y_predict, labels=[0, 1]))
    #
    print("**********************************************************************")
    print("Accuracy on Testing Data :", accuracy)
    print("precision : ", precision)
    print("recall: ", recall)
    print("f1: ", f1)
    print("特征的重要：\n{}".format(tree.feature_importances_))

def RandomForest(x_train, y_train, x_test, y_test):

    '''五颗随机森林'''
    forest = RandomForestClassifier(n_estimators=5, random_state=2)
    forest.fit(x_train, y_train)
    y_predict=forest.predict(x_test)


    # fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    # pltdata = x_train[:, 0:1]
    # print(pltdata)
    # for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):    ax.set_title("Tree {}".format(i))
    # mglearn.plots.plot_tree_partition(pltdata, y_train, tree, ax=ax)
    # mglearn.plots.plot_2d_separator(forest, pltdata, fill=True, ax=axes[-1, -1], alpha=.4)
    # axes[-1, -1].set_title("Random Forest")
    # mglearn.discrete_scatter(pltdata[:, 0], pltdata[:, 1], y_train)

    accuracy = []
    precision = []
    recall = []
    f1 = []
    print('label_true', np.sum(y_test))

    print('label_predict', np.sum(y_predict))

    accuracy.append(metrics.accuracy_score(y_test, y_predict))
    precision.append(metrics.precision_score(y_test, y_predict))
    recall.append(metrics.recall_score(y_test, y_predict))
    f1.append(metrics.f1_score(y_test, y_predict, labels=[0, 1]))
    #
    print("**********************************************************************")
    print("Accuracy on Testing Data :", accuracy)
    print("precision : ", precision)
    print("recall: ", recall)
    print("f1: ", f1)

def GradientTree(x_train, y_train, x_test, y_test):
    #gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
    gbrt = GradientBoostingClassifier(random_state=0,  learning_rate=0.2)
    gbrt.fit(x_train, y_train)
    y_predict=gbrt.predict(x_test)
    print(y_predict)
    print(np.mean(y_predict==y_test))
    accuracy = []
    precision = []
    recall = []
    f1 = []
    print('label_true', np.sum(y_test))

    print('label_predict', np.sum(y_predict))

    accuracy.append(metrics.accuracy_score(y_test, y_predict))
    precision.append(metrics.precision_score(y_test, y_predict))
    recall.append(metrics.recall_score(y_test, y_predict))
    f1.append(metrics.f1_score(y_test, y_predict, labels=[0, 1]))
    #
    print("**********************************************************************")
    print("Accuracy on Testing Data :", accuracy)
    print("precision : ", precision)
    print("recall: ", recall)
    print("f1: ", f1)

def SVMClass(x_train, y_train, x_test, y_test):
    '''
    gamma参数是上一节给出的公式中的参数，用于控制高斯核的宽度。它决定了点与点之间“靠近”是指多大的距离。
    C参数是正则化参数，与线性模型中用到的类似。它限制每个点的重要性（或者更确切地说，每个点的dual_coef_）。

    '''
    svm = SVC(kernel='rbf', C=20, gamma=0.8).fit(x_train, y_train)
    # mglearn.plots.plot_2d_separator(svm, x_train[:, 0], eps=.5)
    # mglearn.discrete_scatter(x_train[:, 0], x_train[:, 1], y_train)
    # # 画出支持向量
    # sv = svm.support_vectors_
    # sv_labels = svm.dual_coef_.ravel() > 0
    # mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
    # plt.xlabel("Feature 0")
    # plt.ylabel("Feature 1")
    # plt.show()

    y_predict = svm.predict(x_test)

    accuracy = []
    precision = []
    recall = []
    f1 = []
    print('label_true', np.sum(y_test))

    print('label_predict', np.sum(y_predict))

    accuracy.append(metrics.accuracy_score(y_test, y_predict))
    precision.append(metrics.precision_score(y_test, y_predict))
    recall.append(metrics.recall_score(y_test, y_predict))
    f1.append(metrics.f1_score(y_test, y_predict, labels=[0, 1]))
    #
    print("**********************************************************************")
    print("Accuracy on Testing Data :", accuracy)
    print("precision : ", precision)
    print("recall: ", recall)
    print("f1: ", f1)



time_start = time.time()
DecisionTree(x_train, y_train, x_test, y_test)
time_end = time.time()
print('DecisionTree time cost', time_end - time_start, 's','\n')

time_start = time.time()
RandomForest(x_train, y_train, x_test, y_test)
time_end = time.time()
print('RandomForest time cost', time_end - time_start, 's','\n')

time_start = time.time()
GradientTree(x_train, y_train, x_test, y_test)
time_end = time.time()
print('GradientTree time cost', time_end - time_start, 's','\n')


time_start = time.time()
#SVMClass(x_train, y_train, x_test, y_test)
time_end = time.time()
print('SVM time cost', time_end - time_start, 's','\n')



