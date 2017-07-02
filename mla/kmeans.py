import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn

from mla.base.base import BaseEstimator
from mla.metrics.distance import euclidean_distance

np.randomseed(1111)


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

    def initialize_centroids(self, init):
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

    def _choose_next_center(self):
        pass
