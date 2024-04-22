#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from cluster import kmeans
#from pca import perform_pca
from sklearn.decomposition import PCA



def perform_pca(data):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    return pca_result


def plot_pca_with_cluster_colors(kmeans_instance, pca_result):
    plt.figure(figsize=(10, 8))
    for i, key in enumerate(kmeans_instance.cluster_dict.keys()):
        print(i)
        cluster_indices = kmeans_instance.cluster_dict[key]
        plt.scatter(pca_result[cluster_indices, 0], pca_result[cluster_indices, 1], label=key)

    plt.title('PCA with Cluster Colors')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig("../examples/pca_++_k=5.png")
    plt.show()

if __name__ == "__main__":
    file = '../data/point5000.lst'
    num_clusters = 5  # Number of clusters as a command line argument
    mykm = kmeans(filename=file, clusters=num_clusters)
    mykm.load(file)
    mykm.cluster()

    # Perform PCA
    pca_result = perform_pca(mykm.data)

    # Plot PCA with cluster colors
    plot_pca_with_cluster_colors(mykm, pca_result)
