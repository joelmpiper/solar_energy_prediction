#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Test the functionality of building the X data from file
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
just_grid = [default_grid]

__, locy = SolarData.load_trainy(trainy_file, loc_file, train_dates,
                                 stations, sta_layout)
trainX, testX = SolarData.make_features(trainX_dir, testX_dir, locy,
                                        train_dates, test_dates, stations,
                                        sta_layout, just_grid)
assert (trainX.loc['2005-11-30', 'ACME']['relative', 'uswrf_sfc', 0, 'SE',
                                         18] == 48)
assert (trainX.loc['2005-11-30', 'ACME']['relative', 'dswrf_sfc', 7, 'SW',
                                         21] == 580)
