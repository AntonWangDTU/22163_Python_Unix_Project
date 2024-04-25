#!/usr/bin/env python3

import sys
import pytest
import numpy as np
testdata_path = '/Users/johan/Documents/UNIXandPython/222110Project/test/testdata'
code_path = '/Users/johan/Documents/UNIXandPython/222110Project/src'
sys.path.append(testdata_path)
sys.path.append(code_path)
from cluster import kmeans

# Making a fixture that will be used to call the kmeans class in all test functions
@pytest.fixture()    
def mykmeans():
    return kmeans()


"""Testing load function in Fasta class"""

# Testing empty file
#def test_load_empty(mykmeans):
#    with pytest.raises(ValueError):
#        mykmeans.load(testdata_path + "empty.txt")

# Testing tab seperated file
def test_tab_sep(mykmeans):
    data, ids = mykmeans.load(testdata_path + "point100_tab.lst")
    assert isinstance(data, np.array) and isinstance(ids, list)


# Testing comma seperated file
#def test_comma_sep(mykmeans):
#    with pytest.raises(ValueError):
#        mykmeans.load(testdata_path + "comma_sep.lst")

# Testing different dimensions
#def test_different_dimensions(mykmeans):
#    with pytest.raises(ValueError):
#        mykmeans.load(testdata_path + "different_dimensions.lst")