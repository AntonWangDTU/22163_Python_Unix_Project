#! /usr/bin/env python3

import sys 
import numpy as np 
import matplotlib.pyplot as plt
from cluster import kmeans 


def calculate_E(kmeans_instance):
    # Calculate the sum-of-squares distances (E) for the current clustering
    E = 0
    for i, key in enumerate(kmeans_instance.cluster_dict.keys()):
        centroid = kmeans_instance.centroids[i]
        for index in kmeans_instance.cluster_dict[key]:
            E += np.sum((kmeans_instance.data[index] - centroid) ** 2)
    return E


def elbow_plot(kmeans_instance, max_clusters):
    # Generate an elbow plot for the given kmeans instance
    E_values = list()
    for num_clusters in range(1, max_clusters + 1):
        print(num_clusters)
        kmeans_instance.clusters = num_clusters
        kmeans_instance.cluster()
        E = calculate_E(kmeans_instance)
        E_values.append(E)   
    plt.plot(range(1, max_clusters + 1), E_values, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Sum-of-Squares (E)')
    plt.title('Elbow Plot: K-means++')
    #plt.title('Elbow Plot: Random') # Uncomment for Random initialization of centroids
    #For adding data to the points
    for i, E in enumerate(E_values):
            E_sci = '{:.2e}'.format(E)
            plt.annotate(f'{E_sci}', (i + 1, E), textcoords="offset points", xytext=(0, 5), ha='center')
    #plt.savefig("<path/to/save/folder/plot_name.png>") # Uncomment and provide path and name to save the plot as a .png file
    plt.show()
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        filename = input("Please enter a data file: ")
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        sys.stderr.write("Usage: elbow_extension.py <datafilename> \n")
        sys.exit(1)
    # Running kmeans algorithm with provided data
    my_kmeans = kmeans()
    my_kmeans.load(filename)
    # Generate elbow plot
    elbow_plot(my_kmeans, max_clusters=10)  # Change max_clusters as needed
