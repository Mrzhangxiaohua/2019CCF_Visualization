import numpy as np
import time
import tensorflow as tf
import random
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
import matplotlib as mpl

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

filterFeatures = pd.read_csv('node2vec1.csv',header=0)
sample = filterFeatures.to_numpy()
print(' * get sample', sample.shape)

ts = time.time()
x2_sample = TSNE(perplexity = 50,n_components=2).fit_transform(sample)
te = time.time()

print('* t-SNE fit cost', (te - ts))
fig, ax = plt.subplots(1, 1)
ax.scatter(x2_sample[:, 0], x2_sample[:, 1], s=0.5)
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_aspect('equal')
plt.show()

kmeans_model4 = KMeans(n_clusters=9, random_state=1)
y_pred = kmeans_model4.fit_predict(x2_sample)
plt.scatter(x2_sample[:, 0], x2_sample[:, 1], c=y_pred)
plt.savefig("pic4.png", bbox_inches='tight')

temp = pd.read_csv('node2vec.csv')
all = pd.concat([temp, pd.DataFrame(kmeans_model4.predict(x2_sample))], axis= 1)
all.to_csv('kmeans.csv')
print("模型聚类评价:" + str(metrics.calinski_harabaz_score(x2_sample, y_pred)))
