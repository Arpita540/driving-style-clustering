# 🚗 Advanced Driving Style Clustering (Single File)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# ================================
# 📂 LOAD DATA
# ================================
df = pd.read_csv("combined_dataset.csv")
df = df.sort_values("window_id")

# ================================
# 🧠 PREPARE FEATURES
# ================================
X = df.drop(["label", "window_id"], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 🔍 FIND BEST K
# ================================
best_k = 2
best_score = -1

for k in range(2,6):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K={k}, Score={score}")

    if score > best_score:
        best_k = k
        best_score = score

print("Best K:", best_k)

# ================================
# 🤖 KMEANS
# ================================
kmeans = KMeans(n_clusters=best_k, random_state=42)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

print("\nKMeans vs Labels:")
print(pd.crosstab(df["kmeans_cluster"], df["label"]))

# ================================
# 🤖 DBSCAN
# ================================
db = DBSCAN(eps=0.5, min_samples=5)
df["dbscan_cluster"] = db.fit_predict(X_scaled)

# ================================
# 🤖 HIERARCHICAL
# ================================
hc = AgglomerativeClustering(n_clusters=best_k)
df["hierarchical_cluster"] = hc.fit_predict(X_scaled)

# ================================
# 📊 PCA VISUALIZATION
# ================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[:,0], X_pca[:,1], c=df["kmeans_cluster"])
plt.title("KMeans PCA")
plt.show()

# ================================
# 📊 TSNE VISUALIZATION
# ================================
tsne = TSNE(n_components=2, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=df["kmeans_cluster"])
plt.title("t-SNE")
plt.show()

# ================================
# 📊 HEATMAP
# ================================
cluster_means = df.groupby("kmeans_cluster").mean(numeric_only=True)

plt.figure()
sns.heatmap(cluster_means, annot=True)
plt.title("Cluster Heatmap")
plt.savefig("plot.png")
plt.close()

# ================================
# 💾 SAVE OUTPUT
# ================================
df.to_csv("final_output.csv", index=False)

print("✅ Done! Output saved as final_output.csv")
