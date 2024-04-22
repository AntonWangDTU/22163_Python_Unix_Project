#!/usr/bin/env python3
import sys 
import numpy as np 
import re
import matplotlib.pyplot as plt



from cluster import kmeans 




def calculate_ssd(kmeans_instance):
    ssd = 0
    for i, key in enumerate(kmeans_instance.cluster_dict.keys()):
        centroid = kmeans_instance.centroids[i]
        for index in kmeans_instance.cluster_dict[key]:
            ssd += np.sum((kmeans_instance.data[index] - centroid) ** 2)
    return ssd

def elbow_plot(kmeans_instance, max_clusters):
    ssd_values = []
    for num_clusters in range(1, max_clusters + 1):
        print(num_clusters)
        kmeans_instance.clusters = num_clusters
        kmeans_instance.cluster()
        ssd = calculate_ssd(kmeans_instance)
        ssd_values.append(ssd)
    
    plt.plot(range(1, max_clusters + 1), ssd_values, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of Squared Distances (SSD)')
    plt.title('Elbow Plot: Random')

    #For adding data to the points
    for i, ssd in enumerate(ssd_values):
            ssd_sci = '{:.2e}'.format(ssd)
            plt.annotate(f'{ssd_sci}', (i + 1, ssd), textcoords="offset points", xytext=(0, 5), ha='center')


    plt.savefig('../examples/elbow_random.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    file = '../data/point1000.lst'
    mykm = kmeans()
    mykm.load(file)
    elbow_plot(mykm, max_clusters=10)  # Change max_clusters as needed