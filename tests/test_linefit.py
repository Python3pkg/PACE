#!/usr/bin/env python3

import pytest
import numpy as np
import PACE

class TestDataStore(object):
    @pytest.fixture
    def data(self):
        return {'__header__':'sadfkljasdfasf',
                '__globals__':[],
                '__version__':'4.2342.42.5.22'}

    @pytest.fixture
    def datastore(self, data):
        testdata = analyze.DataStore('testdatastore.mat')
        testdata.data = data
        return testdata

    def test_get_keys(self, datastore):
        assert(datastore.get_keys() == [])

class TestLineFit(object):
    @pytest.fixture
    def fit(self):
        analyze.LineFit('testfile.mat')

    def test_ddiff(self, fit):
        a = np.array([1, 2, 3])
        assert(analyze.LineFit.ddiff(a) == np.array([1, 1, 1]))
