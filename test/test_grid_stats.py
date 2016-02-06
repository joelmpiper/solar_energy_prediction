#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Test the functionality of creating X data from station names and the
grid variables.
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

var = ['uswrf_sfc', 'dswrf_sfc']

# Keep model 0 (the default model) as a column for each of the variables
# (aggregated over other dimensions)
model = [0, 7]

# Aggregate over all times
times = [18, 21]

default_grid = {'type': 'relative', 'axes': {'var': var, 'models': model,
                                             'times': times,
                                             'station': stations}}

# This just uses the station_names as another feature
stat_names = {'type': 'station_names'}
grid_stats_feats = [default_grid, stat_names]

trainX, trainy, testX, locy = SolarData.load(
    trainX_dir, trainy_file, testX_dir, loc_file, train_dates, test_dates,
    stations, sta_layout, grid_stats_feats)

assert(trainX.loc['2005-11-30', 'ACME']['stat_ACME'] == 1)
assert(trainX.loc['2006-01-01', 'BEAV']['stat_ACME'] == 0)
assert (trainX.loc['2005-11-30', 'ACME']['relative', 0, 'uswrf_sfc', 'SE',
                                         18] == 48)
assert (trainX.loc['2005-11-30', 'ACME']['relative', 7, 'dswrf_sfc', 'SW',
                                         21] == 480)
