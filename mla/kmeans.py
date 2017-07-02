import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

from mla.base.base import BaseEstimator
from mla.metrics.distance import euclidean_distance

random.seed(1111)


class KMeans(BaseEstimator):
    """
    Partition a data set into K clusters.

    Find clusters by repeatedly assigning each data point to the cluster with the closet centroid and iterating until
    the cluster assignments converge (i.e. a data point doesn't change which centroid it is closest to).

    :param k: integer representing the number of clusters to find.
    :param max_iters: maximum number of iterations to perform clustering.
    :param init: a string representing a function name used to initialize clustering. default value is 'random'. Other
    values include:
        'random' - randomly select data points from the data set as the centroids.
        '++' - select a random first centroid from the data set, then select k - 1 centroids by choosing values from
        the data set with a probability distribution proportional to the squared distance from each point's closest
        existing cluster. Attempts to create larger distances between initial clusters to improve convergence rates.
    """

    y_required = False

    def __init__(self, k, max_iters=100, init='random'):
        self.k = k
        self.max_iters = max_iters
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []
        self.init = init

    def _initialize_centroids(self, init):
        """Initialize the centroids"""
        if init == 'random':
            self.centroids = [self.x[y] for y in
                              random.sample(range(self.n_samples), self.k)]
        elif init == '++':
            self.centroids = [random.choice(self.x)]
            while len(self.centroids) < self.k:
                self.centroids.append(self._choose_next_center())
        else:
            raise ValueError('Unknown init parameter: %s' % init)

    def _predict(self, x=None):
        """Perform clustering on the data set"""
        self._initialize_centroids(self.init)
        centroids = self.centroids

        # optimize clusters
        for _ in range(self.max_iters):
            self._assign(centroids)
            centroids_old = centroids
            centroids = [self._get_centroid(cluster) for cluster in self.clusters]

            if self._is_converged(centroids_old, centroids):
                break

        self.centroids = centroids

        return self._get_predictions()

    def _get_predictions(self):
        predictions = np.empty(self.n_samples)

        for i, cluster in enumerate(self.clusters):
            for index in cluster:
                predictions[index] = i

        return predictions

    def _assign(self, centroids):

        for row in range(self.n_samples):
            for i, cluster in enumerate(self.clusters):
                if row in cluster:
                    self.clusters[i].remove(row)
                    break

            closest = self._closest(row, centroids)
            self.clusters[closest].append(row)

    def _closest(self, fpoint, centroids):
        """Find the closest centroid for a point"""
        closest_index = None
        closest_distance = None
        for i, point in enumerate(centroids):
            dist = euclidean_distance(self.x[fpoint], point)
            if closest_index is None or dist < closest_distance:
                closest_index = i
                closest_distance = dist
        return closest_index

    def _get_centroid(self, cluster):
        """Get value by indices and take the mean"""
        return [np.mean(np.take(self.x[:, i], cluster)) for i in range(self.n_features)]

    def _distance_from_centers(self):
        """Calculate distance from centers."""
        return np.array([min([euclidean_distance(x, c) for c in self.centroids]) for x in self.x])

    def _choose_next_center(self):
        distances = self._distance_from_centers()
        probs = distances / distances.sum()
        cumprobs = probs.cumsum()
        r = random.random()
        ind = np.where(cumprobs >= r)[0][0]
        return self.x[ind]

    def _is_converged(self, centroids_old, centroids):
        """Check if the distance between old and new centroids is zero"""
        distance = 0
        for i in range(self.k):
            distance += euclidean_distance(centroids_old[i], centroids[i])
        return distance == 0.0

    def plot(self, ax=None, holdon=None):
        sns.set(style='white')

        data = self.x

        if ax is None:
            _, ax = plt.subplots()

        for i, index in enumerate(self.clusters):
            point = np.array(data[index]).T
            ax.scatter(*point, c=sns.color_palette("hls", self.k + 1)[i])

        for point in self.centroids:
            ax.scatter(*point, marker='x', linewidths=10)

        if not holdon:
            plt.show()