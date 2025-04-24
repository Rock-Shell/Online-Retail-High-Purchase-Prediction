# Modelling
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

scaler = StandardScaler()


def elbow_method(X, k_range=range(1, 20)):
    inertia = []
    scaled_features = scaler.fit_transform(X)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid()
    plt.show()

def train_model(X):
    df2 = X.copy()
    scaled_features = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=7, random_state=42)
    df2['Cluster'] = kmeans.fit_predict(scaled_features)

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=df2['OrderValue'], 
        y=df2['TotalOrders'], 
        hue=df2['Cluster'],
        palette='Set2'
    )
    plt.title('Customer Segments based on OrderValue & TotalOrders')
    plt.show()