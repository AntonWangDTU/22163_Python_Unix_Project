#!/usr/bin/env python3

import sys
import pytest
testdata_path = '/Users/johan/Documents/UNIXandPython/222110Project/test/testdata'
code_path = '/Users/johan/Documents/UNIXandPython/222110Project/src'
sys.path.append(testdata_path)
sys.path.append(code_path)
from cluster import kmeans

# Making a fixture that will be used to call the kmeans class in all test functions
@pytest.fixture()    
def my_kmeans():
    return kmeans()


"""Testing load function in Fasta class"""

# Testing empty file
def test_load_empty(my_kmeans):
    with pytest.raises(ValueError):
        my_kmeans.load(testdata_path + "empty.txt")

# Testing tab seperated file

# Testing comma seperated file
def test_comma_sep(my_kmeans):
    with pytest.raises(ValueError):
        my_kmeans.load(testdata_path + "comma_sep.lst")

# Testing different dimensions
def test_different_dimensions(my_kmeans):
    with pytest.raises(ValueError):
        my_kmeans.load(testdata_path + "different_dimensions.lst")