import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#1. Tao du lieu mau
np.random.seed(42)
X = np.random.randn(300, 2)

#2. Chuan hoa du lieu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#3. Khoi tao va huan luyen mo hinh
# kmeans = KMeans(n_clusters=3, random_state=42)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

#4. Lay nhan va tam cum
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

#5. Visualization
plt.figure(figsize=(15, 5))

#Plot 1 : Ket qua clustering
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:,1], c=labels, cmap='viridis')
plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='x', s=200, linewidths=3)
plt.title('K-means Clustering Results')
plt.colorbar(scatter)

#Plot 2: Elbow Method
inertias = []
K = range(1, 10)

for k in K:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)

plt.subplot(1,2,2)
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Ineritia')
plt.title('Elbow Method For Optimal k')

plt.tight_layout()
plt.show()

#6. In thong tin ve cac cluster
for i in range(3):
    print(f"\nCluster {i}:")
    print(f"Number of points: {np.sum(labels == i)}")
    print(f"Centroid: {centroids[i]}")

#7. Tinh toan va in cac metrics
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(X_scaled, labels)
sse = kmeans.inertia_

print(f'\nModel Metrics: ')
print(f'Silhouette Score: {silhouette_avg: .3f}')
print(f'Sum of Squared Errors: {sse:.3f}')