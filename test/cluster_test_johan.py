#!/usr/bin/env python3

import sys
import pytest
import numpy as np
testdata_path = '/Users/johan/Documents/UNIXandPython/222110Project/test/testdata/'
code_path = '/Users/johan/Documents/UNIXandPython/222110Project/src/'
sys.path.append(testdata_path)
sys.path.append(code_path)
from cluster import kmeans

# Making a fixture that will be used to call the kmeans class in all test functions
@pytest.fixture()    
def mykmeans():
    return kmeans()


"""Testing load function in kmeans class"""

# Testing if correct ValueError is raised when file is empty
def test_load_empty(mykmeans):
    with pytest.raises(ValueError, match="File is empty"):
        mykmeans.load(testdata_path + "empty.txt")

# Testing valid tab seperated file
def test_load_valid_tab_file(mykmeans):
    data, ids = mykmeans.load(testdata_path + "point100_tab.lst")
    assert isinstance(data, np.ndarray) and isinstance(ids, list)
    assert len(data) == 100  # 100 points in the file
    assert len(ids) == 100  # Each point has an ID

# Testing valid comma seperated file
def test_load_valid_comma_file(mykmeans):
    data, ids = mykmeans.load(testdata_path + "point100_comma.lst")
    assert isinstance(data, np.ndarray) and isinstance(ids, list)
    assert len(data) == 100  # 100 points in the file
    assert len(ids) == 100  # Each point has an ID

# Testing valid comma seperated file with no IDs
def test_load_valid_comma_no_ids(mykmeans):
    data, ids = mykmeans.load(testdata_path + "point100_comma_noid.lst")
    assert isinstance(data, np.ndarray) and isinstance(ids, list)
    assert len(data) == 100  # 100 points in the file
    assert len(ids) == 0  # No IDs in the file

# Testing if correct ValueError is raised when vector dimensions are inconsistent
def test_load_different_dimensions(mykmeans):
    with pytest.raises(ValueError, match="Inconsistent vector dimensions"):
        mykmeans.load(testdata_path + "different_dimensions.lst")

# Testing file with one column/dimension
def test_load_one_column(mykmeans):
    data, ids = mykmeans.load(testdata_path + "point100_comma.lst")
    assert isinstance(data, np.ndarray) and isinstance(ids, list)
    assert len(data) == 100  # 100 points in the file
    assert len(ids) == 100  # Each point has an ID

# Testing file with one row (does not make sense to have a file with one row for clustering)
def test_load_one_row(mykmeans):
    data, ids = mykmeans.load(testdata_path + "point1_one_row.lst")
    assert isinstance(data, np.ndarray) and isinstance(ids, list)
    assert len(data) == 1 # 1 point in the file
    assert len(ids) == 1 # Each point has an ID

# Testing file with missing row (100 rows in file, but only 99 points in the data array)
def test_load_missing_row(mykmeans):
    data, ids = mykmeans.load(testdata_path + "point100_missing_row.lst")
    assert isinstance(data, np.ndarray) and isinstance(ids, list)
    assert len(data) == 99
    assert len(ids) == 99



"""Testing cluster function in kmeans class"""

# Testing if correct ValueError is raised when number of clusters exceeds number of observations
def test_cluster_too_many_clusters(mykmeans):
    mykmeans.load(testdata_path + "point100_tab.lst")
    mykmeans.clusters = 101
    with pytest.raises(ValueError, match=f"Number of clusters must not exceed number of obervations which is: {len(mykmeans.vector_list)}"):
        mykmeans.cluster()

# Testing if correct ValueError is raised when number of clusters is less than 1
def test_cluster_less_than_one_cluster(mykmeans):
    mykmeans.load(testdata_path + "point100_tab.lst")
    mykmeans.clusters = 0
    with pytest.raises(ValueError, match="Number of clusters must be greater than 0"):
        mykmeans.cluster()

# Testing if correct ValueError is raised when number of clusters is None
def test_cluster_none_clusters(mykmeans):
    mykmeans.load(testdata_path + "point100_tab.lst")
    with pytest.raises(ValueError, match="Number of clusters must be provided"):
        mykmeans.cluster()

# Testing if correct ValueError is raised when number of clusters is not an integer
def test_cluster_not_integer_clusters(mykmeans):
    mykmeans.load(testdata_path + "point100_tab.lst")
    mykmeans.clusters = 2.5
    with pytest.raises(ValueError, match="Number of clusters must be an integer"):
        mykmeans.cluster()

# Testing if correct ValueError is raised if data has not been loaded
def test_cluster_no_data(mykmeans):
    with pytest.raises(ValueError, match="No data loaded"):
        mykmeans.cluster()



"""Testing write function in kmeans class"""

# Testing if correct ValueError is raised when no clusters have been created
def test_write_no_clusters(mykmeans):
    mykmeans.load(testdata_path + "point100_tab.lst")
    with pytest.raises(ValueError, match="No clusters have been assigned. Please run the cluster method before writing to file."):
        mykmeans.write(testdata_path + "output.lst")