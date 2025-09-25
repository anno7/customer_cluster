# customer_cluster

## 库导入
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyexpat import features
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False
```

## 数据生成
```
np.random.seed(42)
n = 2000

data = {
    "customer_id": range(1,n+1),
    "average_balance":np.random.lognormal(size=n, sigma = 1, mean = 8.5),
    "transaction_frequency":np.random.poisson(lam = 5, size = n),
    "investment_product_count":np.random.choice([0,1,2,3],size = n,p=[0.1,0.2,0.3,0.4]),
}

df = pd.DataFrame(data)

df["log_balance"] = np.log(df["average_balance"])

print("数据生成完毕")
df.shape
```
## 预处理
```features = ["log_balance","transaction_frequency","investment_product_count"]
X = df[features]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("标准化完成")
X_scaled[:5]
```
## K值计算
### 肘部法则
```# 方法1：肘部法则（Elbow Method）
inertias = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)  # 每个簇内点到中心的距离平方和

# 绘制肘部图
plt.figure(figsize=(10, 4))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号（否则负号会变成方块）

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('聚类数量 k')
plt.ylabel('簇内平方和 (Inertia)')
plt.title('肘部法则选择 k')
plt.grid(True)

```
### 轮廓系数
```
# 方法2：轮廓系数（Silhouette Score）
silhouette_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('聚类数量 k')
plt.ylabel('轮廓系数')
plt.title('轮廓系数选择 k')
plt.grid(True)

plt.tight_layout()
plt.show()
```
## 模型训练
```
# 训练最终模型
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 查看每类有多少人
print("各类客户数量：")
print(df['cluster'].value_counts().sort_index())

# 查看每类的平均特征值（用于解释每个群体）
cluster_summary = df.groupby('cluster')[['average_balance', 'transaction_frequency', 'investment_product_count']].mean()
print("\n各类客户平均特征：")
print(cluster_summary.round(2))
```
## 可视化
```
# --- 可视化优化：清晰区分三类客户 ---
plt.figure(figsize=(10, 7))

# 设置高对比度颜色（Set1 有 9 种高区分度颜色）
colors = ['#FF5733', '#33A1FF', '#33FF57']  # 红（高价值）、蓝（活跃）、绿（普通）
cluster_names = ['高净值活跃客户', '活跃理财客户', '普通低频客户']

# 获取每个簇的数据
for i, (name, color) in enumerate(zip(cluster_names, colors)):
    cluster_data = df[df['cluster'] == i]
    plt.scatter(
        cluster_data['average_balance'],
        cluster_data['transaction_frequency'],
        c=color,
        label=name,
        s=60,           # 点的大小
        alpha=0.7,      # 透明度，避免重叠遮挡
        edgecolors='k', # 黑色边框，提升可读性
        linewidth=0.3
    )

# --- 添加聚类中心点（可选，增强专业感）---
centers_scaled = kmeans.cluster_centers_  # 获取标准化后的中心
centers_original = scaler.inverse_transform(
    np.column_stack([
        centers_scaled[:, 0],  # log_balance 还原
        centers_scaled[:, 1],  # transaction_frequency
        centers_scaled[:, 2]   # investment_product_count
    ])
)
# 注意：centers_original[:,0] 是 log_balance，需 exp 还原为原始余额
centers_balance = np.exp(centers_original[:, 0])
centers_freq = centers_original[:, 1]

for i, (name, color) in enumerate(zip(cluster_names, colors)):
    plt.scatter(centers_balance[i], centers_freq[i],
                c=color, s=150, edgecolors='black', linewidth=1.5, marker='X', zorder=10)
    plt.text(centers_balance[i], centers_freq[i] + 0.8,
             f'中心', color=color, fontsize=10, weight='bold',
             ha='center')

# --- 图表美化 ---
plt.xlabel('月均余额（元）', fontsize=12)
plt.ylabel('每月交易次数', fontsize=12)
plt.title('银行客户分群结果（KMeans 聚类）', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')

# --- 图例设置 ---
plt.legend(title='客户群体',
           title_fontsize=11,
           fontsize=10,
           loc='upper right',
           frameon=True,
           fancybox=True,
           shadow=True,
           borderpad=1)

# --- 坐标轴优化 ---
plt.xlim(0, df['average_balance'].max() * 1.05)
plt.ylim(0, df['transaction_frequency'].max() * 1.1)

# --- 显示 ---
plt.tight_layout()
plt.show()
```
