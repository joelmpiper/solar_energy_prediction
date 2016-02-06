#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Test the functionality of building the y data from file
"""

from solar.wrangle.wrangle import SolarData

trainy_file = "solar/data/kaggle_solar/train.csv"
loc_file = "solar/data/kaggle_solar/station_info.csv"
stations = ['ACME', 'BEAV']
sta_layout = True
trainX_dir = "solar/data/kaggle_solar/train/"
testX_dir = "solar/data/kaggle_solar/test/"
train_dates = ['2005-11-30', '2006-01-02']
test_dates = ['2008-12-29', '2009-02-04']
stations = ['ACME', 'BEAV']
sta_layout = True

# This just uses the station_names as another feature
stat_names = {'type': 'station_names'}
stat_feats = [stat_names]

trainX, trainy, testX, locy = SolarData.load(
    trainX_dir, trainy_file, testX_dir, loc_file, train_dates, test_dates,
    stations, sta_layout, stat_feats)

assert(trainX.loc['2005-11-30', 'ACME']['stat_ACME'] == 1)
assert(trainX.loc['2006-01-01', 'BEAV']['stat_ACME'] == 0)
