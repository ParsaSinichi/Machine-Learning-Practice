import numpy as np
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt


def euclidean_distance(point, data):
    return np.sqrt((np.sum((point-data)**2)))


class KMeans:
    def __init__(self, K=5, max_iteration=300,) -> None:
        self.K = K
        self.MaxIter = max_iteration
        self.centroids = []
        self.clusters=[[] for _ in range(self.K)]
    def create_clusters(self, centroids):
        clusters=[[] for _ in range(self.K)]# create a list of list in number of centroids to assign each point to a cluster
        for indx, sample in enumerate(self.X):
            centroid_indx = self.closest_centroid(sample, centroids)
            clusters[centroid_indx].append(indx)# we are appending the index of that point 
            
        return clusters

    def closest_centroid(self, Point):
        """ calculates the distance between each given point and all of the centroids then returns
        the index of the closets Centroid """
        result = [euclidean_distance(Point, Centroid)
                  for Centroid in self.centroids]
        close_indx = np.argmin(result)
        return close_indx
    def get_centroids(self,clusters):
        # assign the man value of clusters to a centroid
        for cluster_indx,cluster in enumerate(clusters):
            np.mean(cluster)
    def fit(self, X):
        self.X = X
        self.nSample, self.nFeature = X.shape
        # Initializing the Centroids :
        # generates K int number in range of nSample, then we use this as an index for sampling from our data
        random_sample = np.random.choice(self.nSample, self.K, replace=False)
        self.centroids = [self.X[index] for index in random_sample]
        # Optimizing :
        for _ in range(self.MaxIter):
            self.clusters=self.create_clusters(self.centroids)


np.random.seed(42)
X, y = make_blobs(centers=3, n_samples=1000, n_features=2,
                  shuffle=True, random_state=40)


# print(y)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
