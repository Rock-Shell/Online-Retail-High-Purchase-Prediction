import pandas as pd
from sklearn.metrics import silhouette_score

def test_prediction(scaled_features, df2):
    silhouette_avg = silhouette_score(scaled_features, df2['Cluster'])
    print("Silhouette Score:", silhouette_avg)
    assert silhouette_avg > 0.5, "Silhouette score is too low, indicating poor clustering."