# K-Means Algorithm Implementation in Python

## Overview

The main goal of this project was to build a Python script with the ability to run the K-means algorithm. As an addition to the standard K-means algorithm, we wanted to make an improvement by using the K-means++ method to pick initial centroids.

We have integrated the K-means clustering algorithm in a class named kmeans for for better modularization and algorithm clarity and to make it possible to import the class in other Python scripts where K-means might be useful. The class consists of:
* One magic method: the instantiation method (_\_init__).
* Three normal methods: loading the data from a file and stores it in a numpy array (load), the K-means clustering method which performs k-means clustering on the loaded data and updates the centroids until convergence is reached (cluster), and a method which writes the cluster assignments and centroids to standard output or to a new file if an output filename is provided (write).
* Four internal methods: calculating the euclidean distance between two datapoints (_euclidean), picking initial centroids from the K-means++ algorithm (\_pick\_centroids\_kmeans\_plusplus), and a method that saves the cluster number in a dictionary where datapoints can get appended (\_initialise\_cluster\_dict). These three internal methods are used inside the cluster function. The fourth internal method is a method that picks random initial centroids (\__pick_centroids_random). However, to use this method, you need to uncomment this line in the cluster function.

The code is located in the "src"-folder with the filename "cluster.py". You can find two Python scripts for plotting the cluster output in a PCA-plot (pca.py) or an elbow plot (SSD.py) in the same folder. In the "test"-folder, you can find the "cluster_test.py"-script which shows the different test, we have used for Unit testing. This testing secures a stable performance of the kmeans class. The data used for testing is in the "testdata" subfolder. The data folder contains examples of data that the kmeans class can handle. Finally, the "examples"-folder shows the plots generated from "pca.py" and "SSD.py".

## Usage

### Installing the Project

The project can be installed directly from Github, e.g., by downloading the page as a ZIP file or cloning using the web URL: https://github.com/AntonWangDTU/222110Project.git

### Data Format

The data should come in a list of data points - either tab or comma separated. The data points may or may not have IDs. Examples:
```
Point00	0.2605	0.6913	0.2874	0.4148
Point01	0.7535	0.1888	0.6069	0.4798
Point02	0.6464	0.4533	0.6959	0.9934
```
```
Point00,0.2605,0.6913,0.2874,0.4148
Point01,0.7535,0.1888,0.6069,0.4798
Point02,0.6464,0.4533,0.6959,0.9934
```
```
0.2605,0.6913,0.2874,0.4148
0.7535,0.1888,0.6069,0.4798
0.6464,0.4533,0.6959,0.9934
```
### Standalone Script

You can run the k-means algorithm as a standalone script by providing a filename of your data and the number of desired clusters as command-line arguments. If you want to save the output in a new file, give the name of the output file as the third argument on the command line. Here is an example:
```
./cluster.py data.lst 3 outfile.lst
```
Here you run the algorithm with "data.lst" as the input data file with 3 clusters and saves the output in a new file called "outfile.lst".

### Module Import

You can also import the kmeans class into other Python scripts and use it as a module. Here's an example which returns the same as in the standalone-example:

```
import sys
code_path = '<insert/path/to/code>'
sys.path.append(code_path)

from cluster import kmeans

my_kmeans = kmeans()
my_kmeans.load("data.lst")
my_kmeans.clusters = 3
my_kmeans.cluster()
my_kmeans.write("outfile.lst")
```

## Requirements

Python3

NumPy

## Authors

Anton Wang Strandberg (s183220)
Contact: s183220@dtu.dk

Johan Filip von Staffeldt (s225001)
Contact: s225001@dtu.dk

## License

This project is licensed under the MIT License.
    
