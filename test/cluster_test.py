#!/usr/bin/env python
import sys

testdata_path = '/home/anton_ws/22110/project/test/testdata'
code_path = '/home/anton_ws/22110/project/src'
sys.path.append(code_path) 
sys.path.append(testdata_path)
import pytest

from cluster import kmeans


#test load funtion within kmeans class

#Test for empty file
def test_load_empty():
    instance = kmeans()
    with pytest.raises(ValueError):
        instance.load("testdata/empty.txt")
#Test for comma seperated file
def test_comma_sep():
    instance = kmeans()
    with pytest.raises(ValueError):
        instance.load("testdata/comma_sep.lst")


