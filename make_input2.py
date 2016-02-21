#!/usr/bin/env python
# coding: utf-8

from solar.wrangle.wrangle import SolarData

# ## Determine all of the parameters

Xtrain_dir = 'solar/data/kaggle_solar/train/'
Xtest_dir = 'solar/data/kaggle_solar/test'
ytrain_file = 'solar/data/kaggle_solar/train.csv'
station_file = 'solar/data/kaggle_solar/station_info.csv'

station = ['all']

train_dates = ['1994-01-01', '1994-06-30']
test_dates = ['2008-01-01', '2008-01-16']

station_layout = True

var = ['all']
model = [0, 7]
times = ['all']
default_grid = {'type': 'relative', 'axes': {
    'var': var, 'models': model, 'times': times, 'station': station}}

stat_names = {'type': 'station_names'}

frac_dist = {'type': 'frac_dist'}

days_solstice = {'type': 'days_from_solstice'}
days_cold = {'type': 'days_from_coldest'}

all_feats = [frac_dist, days_cold, days_solstice, default_grid, stat_names]

all_feats_data = SolarData.load(Xtrain_dir, ytrain_file, Xtest_dir,
                                station_file, train_dates, test_dates,
                                station, station_layout, all_feats, 'local')
