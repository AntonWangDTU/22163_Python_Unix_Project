#!/usr/bin/env python3

import numpy as np
import sys
import re

"""
This is a simple implementation of the k-means algorithm in Python.
The algorithm is implemented in a class called kmeans. The class has
three methods: load, cluster, and write. The load method reads data
from a file and stores it in a numpy array. The cluster method assigns
data points to clusters and updates the centroids. The write method
writes the cluster assignments and centroids to standard output.
The class can be used as a standalone script by providing a filename
and the number of clusters as command-line arguments. The class can
also be imported into other scripts and used as a module.
See README.md for more information.
"""


# Set seed for reproducibility
np.random.seed(42)

class kmeans:
    
    # Initialise the class with the filename and the amount of clusters and needed data structures
    def __init__(self, filename = None, clusters = None):
        self.data = np.array([])
        self.vector_list = list()
        self.ids = list()
        self.clusters = None
        self.cluster_dict = dict()
        self.centroids = None
        if filename is not None:
            self.load(filename)
        if clusters is not None:
            self.clusters = clusters


    def load(self, filename):
        """Reads data from a .lst file and stores it in a numpy array"""
        self.vector_list.clear()
        self.ids.clear()
        try:
            infile = open(filename, "r")
        except IndexError:
            print("Usage: ")
            sys.exit(1)
        except IOError as err:
            print(err)
            sys.exit(1)
        # Read the file and store the data (ids and coordinates/vector) in a list of lists
        for line in infile:
            line = line.split()
            if re.search(r'[A-Za-z]+', line[0]):
                # If the first element is a string, it is an id
                vector = [float(n) for n in line[1:]]
                self.ids.append(line[0])
            else:
                # If the first element is a number, there is no id
                vector = [float(n) for n in line[:]]
            self.vector_list.append(vector)
        infile.close()
        # Convert the list of lists to a numpy array
        cols = len(self.vector_list[0])
        rows = len(self.vector_list)
        self.data = np.empty(shape=(rows, cols), dtype=np.float64)
        for i in range(len(self.vector_list)):
            self.data[i] = self.vector_list[i]
        return self.data, self.ids
    

    def _euclidian(self, v, u):
        '''Calculate the euclidian distance between two vectors'''
        squared_diff = (v - u) ** 2
        distance = np.sqrt(np.sum(squared_diff))
        return distance
    

    def _pick_centroids_kmeans_plusplus(self):
        """Selects the initial centroids using the K-means++ algorithm"""
        # Randomly select the first centroid
        centroid_indices = [np.random.randint(self.data.shape[0])]
        # Choose subsequent centroids using K-means++ algorithm until the desired amount of centroids is reached
        while len(centroid_indices) < self.clusters:
            # Calculate distances from each data point to the nearest centroid
            distances = list()
            for i in range(self.data.shape[0]):
                min_distance = sys.float_info.max
                for centroid_index in centroid_indices:
                    distance = self._euclidian(self.data[i], self.data[centroid_index])
                    min_distance = min(min_distance, distance)
                distances.append(min_distance)
            # Calculate the probability of each data point being selected as the next centroid
            probabilities = np.array(distances) ** 2
            probabilities /= np.sum(probabilities)
            # Select the next centroid based on the calculated probability distribution
            new_centroid_index = np.random.choice(range(self.data.shape[0]), p=probabilities)
            # Append the new centroid index to the list of centroid indices
            centroid_indices.append(new_centroid_index)
        # Create a list of the selected centroids
        initial_centroids = [self.data[i] for i in centroid_indices]
        return initial_centroids
    

    def _initialise_cluster_dict(self):
        '''Initialises the cluster dict according to the designated amount of clusters assigned'''
        self.cluster_dict.clear()
        for i in range(self.clusters):
            self.cluster_dict["Cluster-" + str(i+1)] = list()
        return self.cluster_dict


    def cluster(self):
        '''The function that clusters the data points and updates the centroids until convergence is reached'''     
        convergence = False
        max_iterations = 200
        iteration = 0
        centroids = self._pick_centroids_kmeans_plusplus()
        while not convergence and iteration <= max_iterations:
            new_centroids = list()
            self.cluster_dict = self._initialise_cluster_dict()
            #####---Assign data points to clusters---#####
            # Loop through the observations
            for i in range(self.data.shape[0]):
                distances = list()
                # Loop through centroids and calculate the distance to each centroid
                for j in range(self.clusters):
                    distance = self._euclidian(self.data[i], centroids[j])
                    distances.append(distance)
                # Find the index of the centroid with the lowest distance to the observation
                min_distance = min(distances)
                min_index = distances.index(min_distance)
                # Assign the observation to the cluster with the lowest distance
                cluster_key = "Cluster-" + str(min_index+1)
                self.cluster_dict[cluster_key].append(i)
            #####---Update centroids---#####
            for key in self.cluster_dict.keys():
                # Get the indices of the observations in the cluster
                index_ls = self.cluster_dict[key]
                # Create an empty array of correct shape to store the data points in the cluster
                rows = len(index_ls)
                cols = self.data.shape[1]
                centroid_data = np.empty(shape=(rows, cols), dtype=np.float64)
                # Loop through the indices and store the data points in the cluster
                for i, index in enumerate(index_ls):
                    centroid_data[i] = self.data[index]
                centroid = np.mean(centroid_data, axis = 0)
                # Round the centroid to a certain number of decimals
                centroid = np.round(centroid, decimals=3)
                new_centroids.append(centroid)
            #####---Check for convergence---#####
            if np.array_equal(centroids, new_centroids):
               convergence = True
            # Update the centroids and increase the iteration counter
            iteration += 1
            centroids = new_centroids
        self.centroids = centroids
        return self.cluster_dict, self.centroids
    

    def write(self, outfile):
        '''Writes the cluster assignments and centroids to standard output or writes a new file if outfile is not None'''
        if outfile is None:
            outfile = sys.stdout
        else:
            outfile = open(outfile, "w")
        for i, key in enumerate(self.cluster_dict.keys()):   
            centroids_values = "\t".join([str(value) for value in self.centroids[i]])
            outfile.write(key + "\t" + centroids_values + "\n")
            for index in self.cluster_dict[key]:
                if len(self.ids) == 0:
                    data_point = "\t".join([str(value) for value in self.data[index]])
                    outfile.write(data_point + "\n")
                else:
                    data_point = self.ids[index] + "\t" + "\t".join([str(value) for value in self.data[index]])
                    outfile.write(data_point + "\n")
        if outfile is sys.stdout:
            print("Data was written to standard output. If you want to write to a file, please provide a filename as an argument. Example: mykm.write()")
        else:
            print("Data was written to " + outfile.name)
        outfile.close()

            
     

if __name__ == "__main__":
    # outfilename should be None if it is not given as an argument on the command line
    outfilename = None

    # Get command line arguments
    if len(sys.argv) == 1:
        filename = input("Please enter a data file: ")
        clusters = input("Please enter number of clusters: ")
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        clusters = input("Please enter number of clusters: ")
    elif len(sys.argv) == 3:
        filename = sys.argv[1]
        clusters = sys.argv[2]
    elif len(sys.argv) == 4:
        filename = sys.argv[1]
        clusters = sys.argv[2]
        outfilename = sys.argv[3]
    else:
        sys.stderr.write("Usage: cluster.py <datafilename> <number of clusters> <name of outfile if wanted> \n")
        sys.exit(1)

    # Running kmeans algorithm with provided arguments (data, number of clusters, name of outfile if wanted)
    mykm = kmeans()
    mykm.load(filename)
    mykm.clusters = int(clusters)
    mykm.cluster()
    mykm.write(outfilename)
