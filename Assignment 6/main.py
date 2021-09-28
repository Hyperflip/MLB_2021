import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
import random
from util import nearest_cluster, plot_clusters, is_centroid


class Cluster:
    def __init__(self, centroid):
        self.centroid = centroid    # only features
        self.members = []

    def update_centroid(self):
        if len(self.members) == 0:
            return

        x_values = list(map(lambda x: x[0], self.members))
        y_values = list(map(lambda x: x[1], self.members))

        new_x = 0
        for x in x_values:
            new_x += x
        new_x = new_x / len(x_values)

        new_y = 0
        for y in y_values:
            new_y += y
        new_y = new_y / len(y_values)

        centroid_new = np.array([new_x, new_y])

        self.centroid = centroid_new

    def clear_members(self):
        self.members = []


def k_means(data, k, max_iter):
    # initialize k random clusters
    clusters = []
    for rand_i in random.sample(range(0, k), k):
        rand_sample = data.loc[rand_i]
        cluster = Cluster(rand_sample)          # initialize with centroid

        clusters.append(cluster)

    # do for amount of iterations
    for i in range(0, max_iter):
        # classify each sample by finding nearest cluster by centroid
        for _, sample in data.iterrows():
            nearest_i = nearest_cluster(sample, clusters)
            clusters[nearest_i].members.append(sample)

        plot_clusters(clusters)

        # update centroids and clear members
        convergence = True
        for cluster in clusters:
            centroid_old = cluster.centroid
            cluster.update_centroid()
            if not np.array_equal(centroid_old, cluster.centroid):
                convergence = False
            cluster.clear_members()

        if convergence:
            break


if __name__ == '__main__':
    X, _ = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=2, n_samples=100)
    data = pd.DataFrame(data=X, columns=['X1', 'X2'])

    plt.scatter(data['X1'], data['X2'], marker='o',
                s=25, edgecolor='k')
    #plt.show()

    k = 3
    max_iter = 8
    k_means(data, k, max_iter)
