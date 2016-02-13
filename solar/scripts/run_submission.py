#!/usr/bin/env python
# coding: utf-8

Xtrain_dir = 'solar/data/kaggle_solar/train/'
Xtest_dir = 'solar/data/kaggle_solar/test'
ytrain_file = 'solar/data/kaggle_solar/train.csv'
station_file = 'solar/data/kaggle_solar/station_info.csv'
import solar.wrangle.wrangle
import solar.wrangle.subset
import solar.wrangle.engineer
import solar.analyze.model
import solar.report.submission
import numpy as np


# In[14]:

# Choose up to 98 stations; not specifying a station means to use all that fall within the given lats and longs. If the
# parameter 'all' is given, then it will use all stations no matter the provided lats and longs
station = ['all']

# Determine which dates will be used to train the model. No specified date means use the entire set from 1994-01-01
# until 2007-12-31.
train_dates = ['1994-01-01', '2007-12-31']

#2008-01-01 until 2012-11-30
test_dates = ['2008-01-01', '2012-11-30']

station_layout = True

# Use all variables
var = ['all']

# Keep model 0 (the default model) as a column for each of the variables (aggregated over other dimensions)
model = [0]

# Aggregate over all times
times = ['all']

default_grid = {'type':'relative', 'axes':{'var':var, 'models':model, 'times':times,
                                          'station':station}}
# This just uses the station_names as another feature
stat_names = {'type':'station_names'}

frac_dist = {'type':'frac_dist'}

days_solstice = {'type':'days_from_solstice'}
days_cold = {'type':'days_from_coldest'}

all_feats = [stat_names, default_grid, frac_dist, days_solstice, days_cold]
#all_feats = [stat_names, days_solstice, days_cold]


# In[4]:

import solar.report.submission
import solar.wrangle.wrangle
import solar.wrangle.subset
import solar.wrangle.engineer
import solar.analyze.model

reload(solar.report.submission)
from solar.report.submission import Submission
write = 's3'
station_layout = True

preds = Submission.submit_from_pickle('model_2016-02-06-18-21-41.p',
                                      'input_2016-02-06-18-17-28.p',
                                      station_layout, write)
