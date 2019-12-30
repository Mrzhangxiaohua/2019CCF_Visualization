import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import metrics



def data_process(results):
    # 获取到权重
    print("开始进行处理数据")
    print(len(results))
    up_and_down = []
    # 进行构建边列表
    add_weighted_edges = []
    count = 0
    for result in results:
        # print(result)
        up_and_down.append(int(result["uplink_length"]) + int(result["downlink_length"]))
        count += 1
    all = sum(up_and_down)
    print("所有的数据求和为" + str(all) + str(count))
    count = 0
    for i in range(len(results)):
        add_weighted_edges.append((results[i]["source_ip"], results[i]["destination_ip"], up_and_down[i] / all))
        count += 1
    print("边构造已经完成"+ str(count))

    # 构建图网络表征
    DG = nx.DiGraph()
    DG.add_weighted_edges_from(add_weighted_edges)
    print("图的构建已完成")
    node2vec = Node2Vec(DG, dimensions=2, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node2vec_dict = {}
    for node in DG.nodes():
        node2vec_dict[node] = model.wv[node]
    print(node2vec_dict)
    node2vec_pd = pd.DataFrame.from_dict(node2vec_dict, orient='index')
    node2vec_pd.to_csv('node2vec.csv')
    print("node2vec运算结束")
    print("===========shape")
    print(node2vec_pd.shape)
    kmeans_model = KMeans(n_clusters=5, random_state=1)
    y_pred =  kmeans_model.fit_predict(node2vec_pd)
    print("模型聚类评价:" + str(metrics.calinski_harabaz_score(node2vec_pd, y_pred)))
    kmeans_model1 = KMeans(n_clusters=6, random_state=1)
    y_pred = kmeans_model1.fit_predict(node2vec_pd)
    print("模型聚类评价:" + str(metrics.calinski_harabaz_score(node2vec_pd, y_pred)))
    kmeans_model2 = KMeans(n_clusters=7, random_state=1)
    y_pred = kmeans_model2.fit_predict(node2vec_pd)
    print("模型聚类评价:" + str(metrics.calinski_harabaz_score(node2vec_pd, y_pred)))
    kmeans_model3 = KMeans(n_clusters=8, random_state=1)
    y_pred = kmeans_model3.fit_predict(node2vec_pd)
    plt.savefig("pic3.pdf", bbox_inches='tight')
    print("模型聚类评价:" + str(metrics.calinski_harabaz_score(node2vec_pd, y_pred)))
    kmeans_model4 = KMeans(n_clusters=9, random_state=1)
    y_pred = kmeans_model4.fit_predict(node2vec_pd)
    print("模型聚类评价:" + str(metrics.calinski_harabaz_score(node2vec_pd, y_pred)))
    kmeans_model5 = KMeans(n_clusters=10, random_state=1)
    y_pred = kmeans_model5.fit_predict(node2vec_pd)
    print("模型聚类评价:" + str(metrics.calinski_harabaz_score(node2vec_pd, y_pred)))
    return "ssss"
