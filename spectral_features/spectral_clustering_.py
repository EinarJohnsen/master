"""

Initially taken from:

Author: Ashish Verma
This code was developed to give a clear understanding of what goes behind the
curtains in Spectral clustering.
Feel free to use/modify/improve/etc.

Caution: This may not be an efficient code for production related usage
(especially where data is large) so thoroughly review and test the code
before any usage.
"""

import numpy as np
import scipy
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding, SpectralEmbedding
from sklearn.cluster.k_means_ import k_means


class spectralEmbedding():
    def __Init__(self):
        # variable to store Adjacency matrix
        self.adjacencyMatrix = ""

    def create_distance_matrix(self, X):
        # Based on stackoverflow: https://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy
        s = 1
        pairwise_dists = squareform(pdist(X, 'euclidean'))
        K = scipy.exp(-pairwise_dists ** 2 / (2. * s ** 2))
        self.adjacencyMatrix = K

    def runSpectralEmbedding(self, X, n_components=2, n_clusters=2,
                             k_means_=False):

        # Create distance matrix
        self.create_distance_matrix(X)

        # Run spectral embedding for n_components
        embedding = SpectralEmbedding(n_components=n_components,
                                      affinity='precomputed',
                                      random_state=42,
                                      n_jobs=-1).fit(self.adjacencyMatrix)
        # Alternative way
        # embedding_otherapp = spectral_embedding(self.adjacencyMatrix,
        # n_components=n_components, norm_laplacian=True, random_state=42,
        # drop_first=True)

        # Run k means if set to True
        if k_means_:
            _, kmeans_labels, _ = k_means(X=embedding.embedding_,
                                          n_clusters=n_clusters,
                                          random_state=42, n_init=10)

            # Alternative embedding - More freedom, but slower
            # _, kmeans_labels2, _ = k_means(X=embedding_otherapp, n_clusters=
            # n_clusters, random_state=42, n_init=10)

            return kmeans_labels, embedding.embedding_
        else:
            return embedding.embedding_

    def runSpectralClustering(self, X, n_clusters=2):
        self.create_distance_matrix(X)
        clustering = SpectralClustering(n_clusters=n_clusters,
                                        random_state=42,
                                        affinity='precomputed',
                                        n_jobs=-1).fit(self.adjacencyMatrix)
        return clustering.labels_


# Initialize the class
model = spectralEmbedding()

counter = 0
test_list = []

test_lat_long_list = [[70.361476, 23.891455], [69.868237, 23.320415], [
    69.463576, 17.522170], [69.301095, 20.223625], [68.986478, 22.998791]]
test_lat_long_list_2 = [[59.488506, 11.418907], [60.298552, 9.343399], [
    60.918586, 10.683145], [60.293108, 9.892475], [60.112948, 10.760016]]
#test_lat_long_list_3 = [[58.846615, 5.543791], [58.818186, 5.708514], [
#    59.343169, 5.346124], [59.393549, 5.587717], [59.432681, 6.059923]]

test_lat_long_list_3 = [[58.846615, 5.543791], [58.818186, 5.708514], [
    59.343169, 5.346124], [59.393549, 5.587717], [59.393549, 5.587717]]

lat_long = test_lat_long_list + test_lat_long_list_2 + test_lat_long_list_3

# If shuffle
# np.shuffle(lat_long)

# Spectral embedding
s = model.runSpectralEmbedding(
    lat_long, n_components=2, n_clusters=3, k_means_=False)
print(s)


# Spectral clustering
print(model.runSpectralClustering(lat_long, n_clusters=3))

