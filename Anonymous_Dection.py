from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.manifold import TSNE
import time
import matplotlib as mpl
import matplotlib.pyplot as plt


# X_train = pd.read_pickle("../../../data/flow_used_data/features_all_concatenate.pkl")[965300:969300]
a = pd.read_csv("../../../data/flow_used_data/2019_06_05_process_flow.csv")
a['record_time'] = pd.to_datetime(a['record_time'])
data = a[(a['record_time'] >=pd.to_datetime('2019-06-05 09:00:00')) & (a['record_time'] <= pd.to_datetime('2019-06-05 09:10:00'))]
data_index = data.index.tolist()
X_train = pd.read_pickle("../../../data/flow_used_data/features_all_concatenate.pkl")[data_index]

def Isolation_Forest_function(X_train):
    # 135w数据用时5分钟
    clf = IsolationForest(max_samples='auto', random_state=np.random.RandomState(42), contamination=0.1)
    clf.fit(X_train)
    y_pred_label = clf.predict(X_train).reshape(-1, 1)
    # print(y_pred_label[:20])
    print(np.sum(y_pred_label[y_pred_label[:] == -1]))
    return y_pred_label

def Local_Outlier_Factor_function(X_train):
    #
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    y_pred_label = clf.fit_predict(X_train).reshape(-1, 1)  # Label is 1 for an inlier and -1 for an outlier according to the LOF
    # print(y_pred_label[:20])
    print(np.sum(y_pred_label[y_pred_label[:] == -1]))
    return y_pred_label

def One_ClassSVM_function(X_train):
    clf = svm.OneClassSVM(gamma='auto', nu=0.05)
    clf.fit(X_train)
    y_pred_label = clf.predict(X_train).reshape(-1, 1) #Label is 1 for an inlier and -1 for an outlier according to the LOF、
    print(np.sum(y_pred_label[y_pred_label[:] == -1]))
    print(y_pred_label.shape)
    return y_pred_label

def AE_Result():
    # 目前没有用到
    y_pred_label = pd.read_pickle("AE_labels.pickle").reshape(-1, 1)
    print(y_pred_label[data_index])
    return y_pred_label

def tsne():
    # the default DPI for rendering of `plt.show()` in browser
    mpl.rcParams['figure.dpi'] = 200
    # 减小 bar 的纹理线宽
    mpl.rcParams['hatch.linewidth'] = 0.5
    # bar 的纹理颜色
    mpl.rcParams['hatch.color'] = '#333333'
    # 图的基准字体大小，t
    mpl.rcParams['font.size'] = 7
    mpl.rcParams['axes.facecolor'] = '#FFFFFF'
    mpl.rcParams['figure.facecolor'] = '#FFFFFF'
    mpl.rcParams['figure.edgecolor'] = '#FFFFFF'
    mpl.rcParams['savefig.facecolor'] = '#FFFFFF'
    mpl.rcParams['savefig.edgecolor'] = '#FFFFFF'

    sample = X_train
    print(' * get sample', sample.shape)

    ts = time.time()
    x2_sample = TSNE(perplexity=50, n_components=2).fit_transform(sample)
    te = time.time()

    print('* t-SNE fit cost', (te - ts))
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x2_sample[:, 0], x2_sample[:, 1], s=0.5)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    plt.show()
    return x2_sample


Isolation_Forest_function_label= Isolation_Forest_function(X_train)
Local_Outlier_Factor_function_label = Local_Outlier_Factor_function(X_train)
One_ClassSVM_function_label = One_ClassSVM_function(X_train)
position = tsne()
res = np.concatenate([position, Isolation_Forest_function_label, Local_Outlier_Factor_function_label, One_ClassSVM_function_label],axis=1)
res = pd.DataFrame(res, columns=["X", "Y", "Isolation_Forest_function_label", "Local_Outlier_Factor_function_label", "One_ClassSVM_function_label"])
res.to_csv("../../../data/flow_used_data/2019_06_05_position_and_labels.csv")
print()