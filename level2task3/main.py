import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def load_data(filename='data.csv'):
    df = pd.read_csv(filename)
    return df


def preprocess(df):
    df = df.dropna()
    X = df.select_dtypes(include=['int64', 'float64'])
    return X


def plot_elbow(X_scaled):
    wcss = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        wcss.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.xticks(range(1, 11))
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for K-Means')
    plt.grid(True)
    plt.show()


def plot_silhouette(X_scaled):
    scores = []
    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
        print(f'k = {k}, Silhouette Score = {score:.4f}')

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), scores, marker='o')
    plt.xticks(range(2, 11))
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for K-Means')
    plt.grid(True)
    plt.show()


def fit_kmeans(X_scaled, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    return model, labels


def plot_clusters_pca(X_scaled, labels):
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    df_plot = pd.DataFrame({
        'PC1': components[:, 0],
        'PC2': components[:, 1],
        'Cluster': labels
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_plot,
        x='PC1',
        y='PC2',
        hue='Cluster',
        palette='viridis',
        s=60
    )
    plt.title('K-Means Clustering Visualized with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()


def print_cluster_summary(df, labels):
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = labels
    print('\nCluster counts:')
    print(df_with_clusters['Cluster'].value_counts().sort_index())

    print('\nCluster means:')
    print(df_with_clusters.groupby('Cluster').mean())

    print('\nSample rows by cluster:')
    print(df_with_clusters.groupby('Cluster').head(3))


def print_menu():
    print('\nChoose an option:')
    print('1. Show dataset summary and preview')
    print('2. Plot Elbow Method')
    print('3. Plot Silhouette Scores')
    print('4. Plot K-Means clusters in PCA space')
    print('5. Show cluster summary and centroids')
    print('6. Show key findings')
    print('0. Exit')


def show_data_summary(df):
    print('\nDataset shape:', df.shape)
    print('\nDataset info:')
    print(df.info())
    print('\nMissing values by column:')
    print(df.isnull().sum())
    print('\nNumeric feature preview:')
    print(preprocess(df).head())


def show_key_findings():
    print('\nKey findings:')
    print('- The elbow and silhouette curves help select the optimal number of clusters.')
    print('- Clusters are now grouped in PCA-reduced 2D space for visualization.')
    print('- The cluster summary shows the mean feature values for each segment.')


def main():
    df = load_data('data.csv')
    X = preprocess(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_k = 3

    model = None
    labels = None

    while True:
        print_menu()
        choice = input('Enter option number: ').strip()
        if choice == '0':
            print('Exiting.')
            break
        elif choice == '1':
            show_data_summary(df)
        elif choice == '2':
            plot_elbow(X_scaled)
        elif choice == '3':
            plot_silhouette(X_scaled)
        elif choice == '4':
            if model is None or labels is None:
                model, labels = fit_kmeans(X_scaled, n_clusters=best_k)
                print(f'\nUsing k = {best_k} for final clustering based on elbow and silhouette analysis.')
                print('Final cluster inertia:', model.inertia_)
            plot_clusters_pca(X_scaled, labels)
        elif choice == '5':
            if model is None or labels is None:
                model, labels = fit_kmeans(X_scaled, n_clusters=best_k)
                print(f'\nUsing k = {best_k} for final clustering based on elbow and silhouette analysis.')
                print('Final cluster inertia:', model.inertia_)
            print_cluster_summary(X, labels)
            cluster_centers = scaler.inverse_transform(model.cluster_centers_)
            centers_df = pd.DataFrame(cluster_centers, columns=X.columns)
            centers_df.index.name = 'Cluster'
            print('\nCluster centroids (original feature scale):')
            print(centers_df)
        elif choice == '6':
            show_key_findings()
        else:
            print('Invalid choice. Please select a valid option.')


if __name__ == '__main__':
    main()