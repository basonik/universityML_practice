from typing import Iterable, List
import matplotlib.pyplot as plt
import numpy as np


class AmountClustersError(Exception):
    def __init__(self, n_clusters: int, message="n_clusters can't be negative value") -> None:
        self.n_clusters: int = n_clusters
        self.message: str = message
        super().__init__(message)

    def __str__(self) -> str:
        return f'{self.message} --- n_clusters - {self.n_clusters}'


class KMeans(object):
    """ Классический алгоритм кластеризации KMeans.

    Аргументы конструктора
        n_clusters (int): число кластеров для кластеризации.
    """

    def __init__(self, n_clusters: int, dots: List[Iterable]) -> None:
        if n_clusters <= 0:
            raise AmountClustersError(n_clusters)
        self.n_clusters = n_clusters
        self.dots = dots
        self.centroids = None
        self.clusters = None

    def kmeans_pp(self, X, n_clusters=3, seed=0):
        """Инициализация центроидов методом kmeans++.

        Аргументы:
            X (np.ndarray): Данные для кластеризации, shape = (n_samples, n_features)
            n_clusters (int): Количество кластеров.
            seed (int): random seed.

        Возвращает:
            centroids (np.ndarray): Начальные центроиды (n_clusters, n_features)
        """
        np.random.seed(seed)
        centroids = []
        centroid_ind = np.random.choice(X.shape[0])
        centroids.append(X[centroid_ind])
        for _ in range(n_clusters - 1):
            dists = np.array([min(np.linalg.norm(x - c)**2 for c in centroids) for x in X])
            probs = dists / np.sum(dists)
            next_centroid_idx = np.random.choice(len(X), p=probs)
            centroids.append(X[next_centroid_idx])
        return np.stack(centroids)

    def fit(self, X: np.ndarray = None, min_d: float = 1e-4, max_iter: int = 1000, seed: int = 0) -> 'KMeans':
        """
        Запускает основной алгоритм поиска кластеров.

        Аргументы
            X (np.ndarray) — данные для кластеризации, shape = (n_samples, n_features)
            min_d (float) — минимальное изменение центроидов для останова
            max_iter (int) — максимальное количество итераций
            seed (int) — random seed

        Возвращает
            self
        """
        if X is not None:
            self.dots = X
        elif self.dots is not None:
            X = self.dots
        else:
            raise ValueError("No data provided for fit.")

        np.random.seed(seed)
        n_samples = X.shape[0]
        self.centroids = self.kmeans_pp(X, self.n_clusters, seed)
        
        shift = float('inf')
        iter_count = 0
        while shift > min_d and iter_count < max_iter:
            clusters = [[] for _ in range(self.n_clusters)]
            for x in X:
                dists = np.linalg.norm(self.centroids - x, axis=1)
                cluster_ind = np.argmin(dists)
                clusters[cluster_ind].append(x)
            new_centroids = []
            for j in range(self.n_clusters):
                if clusters[j]:
                    new_centroids.append(np.mean(clusters[j], axis=0))
                else:
                    new_centroids.append(self.centroids[j])
            new_centroids = np.array(new_centroids)
            
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            iter_count += 1

        self.clusters = clusters
        return self

    def predict(self, new_x) -> List[int]:
        """
        Прогнозирует к какому кластеру принадлежат новые точки X.

        Аргументы:
            X (np.ndarray): Точки для предсказания, shape = (n_samples, n_features)

        Возвращает:
            labels (List[int]): номера кластеров для каждой точки
        """
        new_x = np.asarray(new_x)
        labels = []
        for x in new_x:
            dists = np.linalg.norm(self.centroids - x, axis=1)
            cluster_ind = np.argmin(dists)
            labels.append(cluster_ind)
        return labels

    @staticmethod
    def plot_dots(centers, dots) -> None:
        """Рисуем точки и их центроиды. Один кластер — один цвет."""
        colors = plt.cm.tab10.colors
        for i, cluster_points in enumerate(dots):
            cluster_points = np.array(cluster_points)
            color = colors[i % len(colors)]
            if cluster_points.size != 0:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                           color=color, label=f'Кластер {i+1}', alpha=0.6)
            plt.scatter(
                centers[i][0], centers[i][1],
                color=color, marker='o', edgecolor='k',
                s=200, linewidths=2, zorder=10
            )
        plt.legend()
        plt.title('Кластеры и центроиды')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


if __name__ == '__main__':
    np.random.seed(1000)
    n_clusters = 4
    points = 100
    centers = np.random.uniform(-10, 100, (n_clusters, 2))
    clouds = []
    for center in centers:
        cloud = center + np.random.randn(points, 2) * 4
        clouds.append(cloud)
    X = np.concatenate(clouds)

    model = KMeans(n_clusters=4, dots=X)
    model.fit()
    KMeans.plot_dots(model.centroids, model.clusters)