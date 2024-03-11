import numpy as np
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt


def euclidean_distance(point, data):
    return np.sqrt((np.sum((point-data)**2)))


class KMeans:
    def __init__(self, K=5, max_iteration=300, plot_steps=True) -> None:
        self.K = K
        self.MaxIter = max_iteration
        self.centroids = []
        self.clusters = [[] for _ in range(self.K)]
        self.plot_steps = plot_steps

    def get_cluster_labels(self, clusters):
        # creating an empty array in size of our samples
        labels = np.empty(self.nSample)
        for cluster_indx, cluster in enumerate(clusters):
            for sample_indx in cluster:
                labels[sample_indx] = cluster_indx

        return labels

    def create_clusters(self, centroids):
        # create a list of list in number of centroids to assign each point to a cluster
        clusters = [[] for _ in range(self.K)]
        for indx, sample in enumerate(self.X):
            centroid_indx = self.closest_centroid(sample, centroids)
            # we are appending the index of that point
            clusters[centroid_indx].append(indx)

        return clusters

    def closest_centroid(self, Point, Centroids):
        """ calculates the distance between each given point and all of the centroids then returns
        the index of the closets Centroid """
        result = [euclidean_distance(Point, Centroid)for Centroid in Centroids]
        close_indx = np.argmin(result)
        return close_indx

    def get_centroids(self, clusters):
        # assign the man value of clusters to a centroid
        centroids = np.zeros((self.K, self.nFeature))
        for cluster_indx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_indx] = cluster_mean
        return centroids

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

    def is_converged(self, old_cent, cetroids):
        distances = [euclidean_distance(
            old_cent[i], cetroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def calc_mean_distance(self):
        """Calculating the average distance from each point to its labeled cluster centroid.

        """
        Centroid_distance = [[] for _ in range(self.K)]
        for i in range(self.K):
            k.clusters[i]
            Centroid_distance[i] = [euclidean_distance(
                k.clusters[i][j], k.centroids[0])for j in range(len(k.clusters[i]))]
        for i in range(self.K):
            print(
                f" average of distances of points belonging to cluster {i} is {np.mean(Centroid_distance[i])}")
            print(
                f" STD of distances of points belonging to cluster {i} is {np.std(Centroid_distance[i])}")

    # euclidean_distance(X[1],k.get_centroids(k.clusters)[k.closest_centroid(X[1],k.centroids)])
        # return Centroid_distance

    def fit(self, X):
        self.X = X
        self.nSample, self.nFeature = X.shape
        # Initializing the Centroids :
        # generates K int number in range of nSample, then we use this as an index for sampling from our data
        random_sample = np.random.choice(self.nSample, self.K, replace=False)
        self.centroids = [self.X[index] for index in random_sample]
        # Optimizing :
        for _ in range(self.MaxIter):
            self.clusters = self.create_clusters(self.centroids)
            if self.plot_steps:
                pass
                self.plot()

            old_centroids = self.centroids
            self.centroids = self.get_centroids(self.clusters)
            if self.is_converged(old_centroids, self.centroids):
                break
            if self.plot_steps:
                pass
                self.plot()
        return self.get_cluster_labels(self.clusters)


np.random.seed(42)
X, y = make_blobs(centers=5, n_samples=1000, n_features=2,
                  shuffle=True, random_state=74, cluster_std=743)


clusters = len(np.unique(y))
k = KMeans(K=clusters, max_iteration=300, plot_steps=True)
y_pred = k.fit(X)

k.plot()
k.calc_mean_distance()
