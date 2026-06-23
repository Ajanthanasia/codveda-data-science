import pandas
import numpy
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

load_data = pandas.read_csv("data.csv")

print(load_data.head())
print(load_data.info())

print(load_data.isnull().sum())

load_data = load_data.dropna()

# print('load data head:')
# print(load_data.head())

# print('load data shape:')
# print(load_data.shape)

# print('load data info:')
# print(load_data.info())

X = load_data.select_dtypes(include=['int64', 'float64'])

print(X.head())
# X = load_data

# scaler = StandardScaler()

# X_scaled = scaler.fit_transform(X)

# inertia = []

# for k in range(1, 11):
#     model = KMeans(
#         n_clusters=k,
#         random_state=42
#     )

#     model.fit(X_scaled)

#     inertia.append(model.inertia_)

# plt.plot(range(1,11), inertia, marker='o')

# plt.xlabel("Number of Clusters")

# plt.ylabel("Inertia")

# plt.title("Elbow Method")

# plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

for k in range(2,11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    print(f"K={k}, Silhouette Score={score:.3f}")
    
    kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)

load_data['Cluster'] = clusters

print(load_data.head())

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_df = pandas.DataFrame(
    X_pca,
    columns=['PC1', 'PC2']
)

pca_df['Cluster'] = clusters

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    pca_df['PC1'],
    pca_df['PC2'],
    c=pca_df['Cluster'],
    cmap='viridis'
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters (PCA)")
plt.colorbar(scatter)
plt.show()

centers = pandas.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=X.columns
)

print(centers)