# 2019CCF企业网络资产及安全事件分析与可视化（https://www.datafountain.cn/competitions/358）

## 企业网络资产及安全事件分析与可视化第二名

### 初赛：资产分类并可视化思路（参考[TSNE.py](./Code/TSNE.py)、[Node2Vec_and_Cluster.py](./Code/Node2Vec_and_Cluster.py)）
    1. 根据资产之间通信关系，得到网络关系
    2. 对于其他信息，如上行、下行流量、端口号等进行特征处理
    3. 根据1和2信息进行网络表征
    4. 使用算法node2vec进行网络表征
    5. 使用TSNE算法降维可视化，观察是否呈现聚类特性
    6. 发现有簇特征，因此是可分的，故使用K-MEANS聚类、并打标
    （也可用其他的聚类方法如谱聚类，个人机器跑不动，所以选取了速度快的）
    7.分析参考答卷

### 复赛：从周期访问突变、流量访问异常等角度找出存在的异常通信模式思路（参考[URI_TFIDF.py](./Code/URI_TFIDF.py)、[Word_Cut.py](./Code/Word_Cut.py)、[Anonymous_Dection.py](./Code/Anonymous_Dection.py)）  
    1. 从不同角度出发，找到异常通信模式（我们找出的有五种异常），主要解释算法发现异常
    2. 针对flow表进行分析（我们查阅论文后，发现脚本注入是一种异常）。
    我们对uri、useragent进行处理，去除停用词等信息，并通过word2vec进行特征编码，
    对method、host等进行onehot编码处理，最终得到处理后的特征向量。
    3. 使用异常检测算法进行检测：1）Isolation_Forest，2）Local_Outlier_Factor，3）One_ClassSVM
    4. 对异常结果进行可视化呈现
    5. 分析参考答卷

### 由于方案赛代码量较大，我选取其中算法部分予以呈现，具体细节参考答卷和论文
参考[初赛-答卷.pdf](./Description/初赛-答卷.pdf)、 [复赛-答卷.pdf](./复赛-答卷.pdf)、[说明论文.docx](./Description/说明论文.docx)
