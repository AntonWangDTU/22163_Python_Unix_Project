#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from cluster import kmeans

"""
This script is an extension of the cluster.py script for better visualization of the clusters.
The script consists of two functions:
1. perform_pca(data): This function performs PCA on the given data and returns the result.
2. plot_pca_with_cluster_colors(kmeans_instance, pca_result): This function plots the data points in the 2D space of the top two principal components with cluster colors.
The script takes the following command line arguments:
1. datafilename: The name of the file containing the data.
2. number of clusters: The number of clusters to be formed.
"""


def perform_pca(data):
    """This function performs PCA on the given data and returns the result."""
    # Step 1: Standardize the data
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    standardized_data = (data - mean) / std_dev
    # Step 2: Calculate the covariance matrix
    covariance_matrix = np.cov(standardized_data.T)
    # Step 3: Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # Step 4: Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    # Step 5: Select the top k eigenvectors
    top_k_eigenvectors = sorted_eigenvectors[:, :2]
    # Step 6: Project the data onto the top k eigenvectors
    pca_result = np.dot(standardized_data, top_k_eigenvectors)
    return pca_result


def plot_pca_with_cluster_colors(kmeans_instance, pca_result):
    """This function plots the data points in the 2D space of the top two principal components with cluster colors."""
    plt.figure(figsize=(10, 8))
    for i, key in enumerate(kmeans_instance.cluster_dict.keys()):
        print(i)
        cluster_indices = kmeans_instance.cluster_dict[key]
        plt.scatter(pca_result[cluster_indices, 0], pca_result[cluster_indices, 1], label=key)
    plt.title('PCA with Cluster Colors')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    #plt.savefig("<path/to/save/folder/plot_name.png>") # Uncomment and provide path and name to save the plot as a .png file
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        filename = input("Please enter a data file: ")
        clusters = input("Please enter number of clusters: ")
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        clusters = input("Please enter number of clusters: ")
    elif len(sys.argv) == 3:
        filename = sys.argv[1]
        clusters = sys.argv[2]
    else:
        sys.stderr.write("Usage: PCA_extension.py <datafilename> <number of clusters> \n")
        sys.exit(1)
    # Running kmeans algorithm with provided arguments (data, number of clusters)
    my_kmeans = kmeans()
    my_kmeans.load(filename)
    if not clusters.isdigit():
        sys.stderr.write("Number of clusters must be an integer\n")
        sys.exit(1)
    my_kmeans.clusters = int(clusters)
    my_kmeans.cluster()
    # Perform PCA
    pca_result = perform_pca(my_kmeans.data)
    # Plot PCA with cluster colors
    plot_pca_with_cluster_colors(my_kmeans, pca_result)
