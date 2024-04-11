#!/usr/bin/env python3

import numpy as np
import sys
import re 



np.random.seed(42)


class kmeans:



    def __init__(self, filename = None, clusters = 3):
        '''clusters: number of clusters stated in command line'''
        self.filename = filename
        self.clusters = clusters




    def load(self, filename):


        try:
            infile = open(filename, "r")
        except IndexError:
            print("Usage: ")
            sys.exit(1)
        except IOError as err:
            print(err)
            sys.exit(1)

        self.vector_list = []
        self.ids = []
        for line in infile:
            line = line.split()
            if re.search(r'[A-Za-z]+', line[0]):
                vector = [float(n) for n in line[1:]]
                self.ids.append(line[0])
            else:
                vector = [float(n) for n in line[:]]
            self.vector_list.append(vector)
        infile.close()

        cols = len(self.vector_list[0])
        rows = len(self.vector_list)

        #Turn list of list into np array
        self.data = np.empty(shape=(rows, cols), dtype=np.float64)

        for i in range(len(self.vector_list)):
            self.data[i] = self.vector_list[i]


        return self.data, self.ids
    
    def _eucledian(self, v, u):
        
        squeared_diff = (v - u) ** 2

        distance = np.sqrt(np.sum(squeared_diff))

        return distance

    def _pick_centroids_random(self):

        centroid_ids = np.random.choice(self.data.shape[0], size = self.clusters, replace = False)
        initial_centroids = [self.data[i] for i in centroid_ids]

        return initial_centroids
    
    def _initalise_cluster_dict(self):
        '''Initalises the cluster dict according to the designated amount
        of clusters assigned'''


        self.cluster_dict = {}

        for i in range(self.clusters):
            self.cluster_dict["Cluster-" + str(i+1)] = []

        return self.cluster_dict

    def cluster(self):
        
      

        
        convergence = False
        max_iterations = 200
        iter = 0

        centroids = self._pick_centroids_random()

        while not convergence and iter <= max_iterations:

            new_centroids = []
            self.cluster_dict = self._initalise_cluster_dict()



            #####---Assign data to clusters---#####
            #Loop through the observations
            for i in range(self.data.shape[0]):

                distances = []

                #Loop through centroids
                for j in range(self.clusters):

                    distance = self._eucledian(self.data[i], centroids[j])
                    distances.append(distance)

                # Find cluster(index) that has the lowest distance
                min_distance = min(distances)
                min_index = distances.index(min_distance)

                #assign the rows indexes to the clusters
                cluster_key = "Cluster-" + str(min_index+1)
                self.cluster_dict[cluster_key].append(i)
        
            

            #####---Update centroids---#####

            for key in self.cluster_dict.keys():


                #The list of indexes assigens to current cluster
                index_ls = self.cluster_dict[key]


                #Shape of array to calculate centroid from 
                rows = len(index_ls)
                cols = self.data.shape[1]


                #make empty np array with shape of the amount of observations in each cluster
                centroid_data = np.empty(shape=(rows, cols), dtype=np.float64)


                #Loop through index_ls and assign data pouts 
                for i, index in enumerate(index_ls):
                    centroid_data[i] = self.data[index]

                
                centroid = np.mean(centroid_data, axis = 0)
                #new_centroids.append(centroid)

                # Round the centroid to a certain number of decimal places
                centroid = np.round(centroid, decimals=3)  # Adjust decimals as needed
                new_centroids.append(centroid)



            if np.array_equal(centroids, new_centroids):
               convergence = True
            iter += 1
            centroids = new_centroids
        
        self.centroids = centroids

    def write(self):
        

        for i, key in enumerate(self.cluster_dict.keys()):
            
            centroids_values = "\t".join([str(value) for value in self.centroids[i]])
            print(key + "\t" + centroids_values)
            
            for index in self.cluster_dict[key]:
                data_point = self.ids[index] + "\t" + "\t".join([str(value) for value in self.data[index]])
                print(data_point)

            
        

if __name__ == "__main__":
    file = sys.argv[1]
    #file = '../data/point100.lst'
    mykm = kmeans()
    mykm.load(file)
    mykm.cluster()
    mykm.write()







