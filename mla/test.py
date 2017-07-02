import random

from sklearn.datasets import make_blobs
from mla.kmeans import KMeans


def kmeans():
    """
    Test the kmeans module. generate a synthetic data set and cluster the data. Plot the results in seaborn.
    """
    n_samples = 1500
    random_state = random.randint(1, 255)
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    kmeans_test = KMeans(k=3)
    kmeans_test.fit(X)
    kmeans_test.predict(X)
    kmeans_test.plot()


if __name__ == '__main__':
    kmeans()