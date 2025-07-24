# Objective:  
To study fundamental concepts of unsupervised learning. As part of the practical work, it is required to:

1. Evaluate theoretical aspects of the k-means clustering algorithm
2. Implement the algorithm using an available programming language


# Implementation Requirements:  
• The practical work involves using Python programming language  
• Utilization of publicly available libraries (e.g., numpy, matplotlib)  
• Quality assessment of the algorithm's performance

# Task Description:
The assignment requires implementing the k-means algorithm using OOP principles in Python. The implementation must include:

- Functions for calculating centroids
- Capability to cluster new data
- Data visualization on a two-dimensional plane




### Introduction

In the era of rapidly growing data volumes and the need for efficient processing, machine learning algorithms have gained significant popularity due to their ability to automate data analysis and uncover hidden patterns. One of the most essential tasks in data analysis is **clustering**-the automatic grouping of objects based on similarity.

This practical work focuses on **k-means**, a fundamental clustering algorithm. The study emphasizes:

- The **theoretical foundations** of the method
- An **object-oriented implementation** in Python using widely adopted libraries (`numpy` and `matplotlib`)

The goals of this work extend beyond introducing unsupervised learning concepts. It also aims to develop practical skills in:

- Algorithm programming
- Data analysis
- Results visualization

Special attention is given to:

- Evaluating the algorithm’s performance
- Investigating its behavior across diverse datasets


### **Main Content**

The **k-means algorithm** is one of the most widely used clustering methods. Its objective is to partition input data into *k* clusters such that:

- Objects within the same cluster are maximally **similar** to each other
- Objects from different clusters are **distinct**

#### **Iterative Process of k-means**:

1. **Initialize** the number of clusters *k*.
2. **Randomly select** *k* cluster centers (centroids):
    - Initial centroids: _μ₁, μ₂, ..., μₖ_
3. **Assign each object** to the nearest centroid:
    Ck={xn:∥xn−μk∥2≤∥xn−μl∥2 ∀ l≠k}Ck​={xn​:∥xn​−μk​∥2≤∥xn​−μl​∥2∀l=k}
    where _Cₖ_ is the set of points belonging to cluster *k*.
4. **Recalculate centroids** as the mean of all points in the cluster:
    μk=1∣Ck∣∑xn∈Ckxnμk​=∣Ck​∣1​xn​∈Ck​∑​xn​
5. **Repeat** steps 3–4 until:
    - Centroids stabilize (no further changes), **or**
    - A stopping criterion is met (e.g., max iterations).



### **k-means++ Algorithm**

The **k-means++** algorithm is an enhancement of the standard k-means method, designed to improve the initialization of cluster centers. In the classical approach, *k* centroids are selected randomly, which may lead to poor initialization and slow convergence. In contrast, **k-means++** selects centroids to maximize their mutual separation, significantly reducing the risk of suboptimal clustering.

#### **Initialization Process (k-means++)**:

1. **Select the first centroid** randomly from the dataset.
2. **Choose each subsequent centroid** from the remaining data points with a probability proportional to the squared distance to the nearest already-selected centroid:
    P(xi)=D(xi)2∑xj∈XD(xj)2P(xi​)=∑xj​∈X​D(xj​)2D(xi​)2​
    where D(xi)D(xi​) is the distance to the nearest centroid.
3. After selecting *k* centroids, proceed with the standard **k-means algorithm**.

---

### **Advantages of k-means**

✔ **Simplicity**: Easy to implement and interpret.  
✔ **Scalability**: Efficient for large datasets (O(nkd)O(nkd) per iteration).  
✔ **Flexibility**: Adapts well to high-dimensional data.

### **Limitations of k-means**

✖ **Cluster count (*k*) must be predefined**.  
✖ **Sensitive to initial centroid positions** (though k-means++ mitigates this).  
✖ **Local optima**: Converges to nearest minima; results may vary across runs.  
✖ **Spherical bias**: Performs poorly on non-convex or irregularly shaped clusters.  
✖ **Noise sensitivity**: Lacks built-in robustness to outliers or noisy data.


### **Conclusion**

This report provided a detailed examination of clustering algorithms, specifically the **k-means** method and its enhanced variant, **k-means++**. The key operational stages of each algorithm were outlined, highlighting their strengths and weaknesses, with particular emphasis on the importance of proper **centroid initialization** for achieving high-quality clustering results.

The analysis demonstrated that **k-means++** significantly improves robustness against poor initial centroid selection, delivering more stable and accurate results compared to standard k-means. However, both methods share inherent limitations:

- The number of clusters (*k*) must be predefined.
- Performance is sensitive to data geometry (e.g., non-spherical clusters).
- Vulnerable to outliers and noise due to the lack of built-in noise-handling mechanisms.

Despite these constraints, **k-means and k-means++ remain widely adopted and effective tools** for diverse clustering tasks. To optimize results, practitioners should:

1. **Preprocess data** (e.g., scaling, outlier removal).
2. **Carefully select *k*** (using metrics like the elbow method or silhouette score).
3. **Evaluate cluster stability** (e.g., via multiple runs with different initializations).


# Listing of code:
```python
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
```
