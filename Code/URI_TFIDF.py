from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import time
import pandas as pd
import numpy as np

def crops_to_tfidf_and_to_uri_feature():
    t1 = time.time()
    x_train = []
    x_test = []
    with open("../data/flow_used_data/crops.txt",'r') as f:
        for line in f:
            x_train.append(line.strip('\n'))
            x_test.append(line.strip('\n'))
    print("---->文件读取完毕")

    vectorizer = CountVectorizer(max_features=20)
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(x_train))
    x_train_weight = tf_idf.toarray()
    print("---->文件训练完毕")

    # 对测试集进行tf-idf权重计算
    tf_idf = tf_idf_transformer.transform(vectorizer.transform(x_test))
    x_test_weight = tf_idf.toarray()  # 测试集TF-IDF权重矩阵
    print("---->文件测试完毕")
    t2 = time.time()
    pd.to_pickle(x_test_weight, "../data/flow_used_data/uri_feature.pkl")
    print("---->文件保存成pkl完毕")
    #
    print('输出x_train文本向量：')
    print(x_train_weight.shape)
    print('输出x_test文本向量：')
    print(x_test_weight.shape)
    print("耗费时间为： ", t2-t1)

def crops_host_to_tfidf_and_to_host_feature():
    t1 = time.time()
    x_train = []
    x_test = []
    with open("../data/flow_used_data/crops_host.txt", 'r') as f:
        for line in f:
            x_train.append(line.strip('\n'))
            x_test.append(line.strip('\n'))
    print("---->文件读取完毕")

    vectorizer = CountVectorizer(max_features=20)
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(x_train))
    x_train_weight = tf_idf.toarray()
    print("---->文件训练完毕")

    # 对测试集进行tf-idf权重计算
    tf_idf = tf_idf_transformer.transform(vectorizer.transform(x_test))
    x_test_weight = tf_idf.toarray()  # 测试集TF-IDF权重矩阵
    print("---->文件测试完毕")
    t2 = time.time()
    pd.to_pickle(x_test_weight, "../data/flow_used_data/host_feature.pkl")
    print("---->文件保存成pkl完毕")
    #
    print('输出x_train文本向量：')
    print(x_train_weight.shape)
    print('输出x_test文本向量：')
    print(x_test_weight.shape)
    print("耗费时间为： ", t2 - t1)

def crops_host_to_tfidf_and_to_host_feature():
    df = pd.read_csv('../data/flow_used_data/2019_06_05_process_flow.csv')['method'].replace([np.nan], 'GET')
    df = pd.Series(df)
    c = pd.get_dummies(df,sparse=True).to_numpy()
    pd.to_pickle(c, "../data/flow_used_data/method_feature.pkl")
    print("---->文件保存成pkl完毕")

# 分别提取特征信息
# crops_to_tfidf_and_to_uri_feature()
# crops_host_to_tfidf_and_to_host_feature()
# crops_host_to_tfidf_and_to_host_feature()


a = pd.read_pickle("../data/flow_used_data/uri_feature.pkl")
b = pd.read_pickle("../data/flow_used_data/host_feature.pkl")
c = pd.read_pickle("../data/flow_used_data/method_feature.pkl")

features_all = np.concatenate([a,b,c],axis=1)
pd.to_pickle(features_all, "../data/flow_used_data/features_all_concatenate.pkl")
print()
