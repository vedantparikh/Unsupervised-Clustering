from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class Model:

    def get_kmeans_silhouette_scores(self, iterations: int, data: np.array, **kwargs) -> Tuple[list, list]:
        """Returns Kmeans and Silhouette scores."""

        wcss = []
        silhouette_coefficients = []
        for i in range(1, iterations):
            kmeans_pca = KMeans(n_clusters=i, **kwargs)
            kmeans_pca.fit(data)
            if not i == 1:
                score = silhouette_score(data, kmeans_pca.labels_)
                silhouette_coefficients.append(score)
            wcss.append(kmeans_pca.inertia_)

        return wcss, silhouette_coefficients

    def train(self, data: np.array, n_clusters: int = 2, **kwargs) -> 'KMeans':
        """Returns a trained KMeans model."""

        kmeans_pca = KMeans(n_clusters=n_clusters, **kwargs)
        kmeans_pca.fit(data)

        return kmeans_pca

    def predict(self, data: np.array, kmeans_pca: 'KMeans') -> np.array:
        """Retrurns model prediction."""
        predictions = kmeans_pca.predict(data)

        return predictions
