
# coding: utf-8

# # Analysis Walkthrough

# ## Determine all of the parameters

# ### Specify the locations of the files for X_train, X_test, and y_train. Also, there is a file that contains information about the individual stations that can be useful for models that learn for each station.

# In[1]:

import os
os.chdir('..')


# In[22]:

Xtrain_dir = 'solar/data/kaggle_solar/train/'
Xtest_dir = 'solar/data/kaggle_solar/test'
ytrain_file = 'solar/data/kaggle_solar/train.csv'
station_file = 'solar/data/kaggle_solar/station_info.csv'


# ### Import the various files. This is mostly done so that any file updates during testing are carried over to this notebook.

# In[23]:

import solar.wrangle.wrangle
import solar.wrangle.subset
import solar.wrangle.engineer
import solar.analyze.model
import solar.report.submission
import numpy as np


# ### Set the parameters that will be used to set up the data. There are some parameters that determine the size and shape of the data but have effects other than setting up feature columns. This includes the dates that are included for testing and training, the stations considered, and whether to have X values correspond to a date or to a specific date/station combination.

# In[26]:

# Choose up to 98 stations; not specifying a station means to use all that fall within the given lats and longs. If the
# parameter 'all' is given, then it will use all stations no matter the provided lats and longs

#station = ['all']
station = ['ACME', 'BEAV']

# Determine which dates will be used to train the model. No specified date means use the entire set from 1994-01-01
# until 2007-12-31.

#train_dates = ['1994-01-01','2007-12-31']
train_dates = ['1994-01-01','1994-03-01']

# Determine the test X values to produce. There is no practical purpose to use fewer than all of the points other than
# for testing. Again, not choosing a date will use 2008-01-01 through 2012-11-30.

#test_dates = ['2008-01-01','2012-11-30']
test_dates = ['2008-01-01', '2008-01-05']

# The last parameter that is not specifically involved in feature selection in the layout to be used for training
# I have switched to almost always training for each individual station rather than having a single row for a date.
# However, I am still not beating the benchmark, and the file would grow too large to have the benchmark laid out
# with a row for each station, so I'll keep the switch. True means that each station has a row (5113 dates X 98
# stations to train the data). False means that there are 5113 rows that are being used to train the data.
station_layout = True


# ### First, just duplicate the functionality of the basic grid analysis

# In[27]:

# Use all variables

#var = ['all']
var = ['uswrf_sfc', 'dswrf_sfc', ]

# Keep model 0 (the default model) as a column for each of the variables (aggregated over other dimensions)
model = [0]

# Aggregate over all times
#times = [12, 15, 18, 21, 24]
times = [21]

default_grid = {'type':'relative', 'axes':{'var':var, 'models':model, 'times':times,
                                          'station':station}}
just_grid = [default_grid]


# ### Run data extraction

# In[28]:

# if I am modifying code for any of these pythons
reload(solar.wrangle.wrangle)
reload(solar.wrangle.subset)
reload(solar.wrangle.engineer)
from solar.wrangle.wrangle import SolarData

get_ipython().magic(u"prun input_data = SolarData.load(Xtrain_dir, ytrain_file, Xtest_dir, station_file,                                   train_dates, test_dates, station,                                   station_layout, just_grid, 'local')")


# In[29]:

input_data[0]


# ### Run through the full analysis

# In[170]:

reload(solar.analyze.model)
import numpy as np
from solar.analyze.model import Model

from sklearn.linear_model import Ridge
from sklearn import metrics

error_formula = 'mean_absolute_error'
cv_splits = 10

model = Model.model(input_data, Ridge, {'alpha':np.logspace(-3,1,8,base=10)}, cv_splits,
                    error_formula, 1, 'local', normalize=True)


# In[ ]:

reload(solar.report.submission)
from solar.report.submission import Submission

preds = Submission.make_submission_file(model, input_data[1], input_data[2], {'grid'})


# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:

# here we set some parameters to set figure size and style
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14
sns.set(style="white", context="talk")


# In[ ]:

y_pred = model.predict(input_data[0])


# In[ ]:

errors = abs(y_pred - input_data[1])/input_data[1]


# In[ ]:

#sns.distplot(errors, rug=True)


# In[ ]:

#input_data[0].shape


# In[ ]:

#errors.shape


# In[ ]:

#input_data[0][(errors > 0.8).values]


# In[ ]:

#input_data[1][(errors > 0.8).values]


# In[ ]:

#pd.DataFrame(y_pred, index=input_data[1].index, columns=input_data[1].columns)[(errors > 0.8).values]


# In[ ]:

#max(errors.values)


# In[ ]:

#unstacked = input_data[0].stack('time').stack('model').stack('variable').stack('lat_longs').reset_index('model').reset_index('time').reset_index('variable').reset_index('lat_longs')


# In[ ]:

#dswrf = unstacked[(unstacked['lat_longs'] == 'NE') & (unstacked['variable'] == 'dswrf_sfc') & (unstacked['time'] == 21)
#         & (unstacked['model'] == 0)]['relative']


# In[ ]:

#pd.DataFrame(dswrf).reset_index('station').rename(columns={'station':'location'}).unstack('location').set_index('location')


# In[ ]:

#dswrf


# In[ ]:

#pd.concat((errors,dswrf))


# In[ ]:

#pd.concat([dswrf,errors])


# In[ ]:

#sns.jointplot("total_solar","dswrf", data=drinks[(drinks.beer < 100) & (drinks.wine < 30)] , kind = "kde")


# In[ ]:

#import netCDF4 as nc
#X = nc.Dataset('solar/data/kaggle_solar/train/dswrf_sfc_latlon_subset_19940101_20071231.nc','r+').variables.values()
#X[-1][0:10,0,2:3,:,:]


# In[6]:

# This just uses the station_names as another feature
stat_names = {'type':'station_names'}
stat_feats = [stat_names]


# ### Next, we start to layout the features to include. The two most important (and complicated) are 'absolute', which just reports out the weather variables at specific GEFS, times, and models, and 'relative' which uses a grid to identify nearby GEFS for weather measurements based on the location of the station. The second option makes the most sense when using the station_layout above, but it will work with either layout.

# In[ ]:

# A very simple model would just take the average value of all variables at all locations, using all models over the
# course of the day. Here, only the var parameter and one value of the model is expanded.
# All of the other axes are aggregated using an aggregation function. In this case, the mean value.
# This will provide a 15 aggregated columns for model 0 and 15 aggregated columns for the mean of models 1 though 10.
# In this case, setting station_layout to false would make the most sense because the measurements will be repeated
# for each station. However, for consistency in this walkthough, I will just keep it in the station_layout.

# Dimensions without aggregation

# Use all variables
var = ['all']

# Keep model 0 (the default model) as a column for each of the variables (aggregated over other dimensions)
model1 = [0]

# Dimensions with aggregation

# Aggregate over all other models (excluding model 0, which is used directly)
model2 = range(1,11)

# Aggregate over all times
times = ['all']

# Aggregate over all latitudes which surround Mesonet stations (exclude those that are outside of the main grid)
lats = range(33,38)

# Same as for lats
longs = range(257,267)

all_avgs = {'type':'absolute', 'full_axes':{'var':var, 'models':model1},
            'agg_axes':{'models':[model2,[np.mean]], 'times':[times, [np.mean, np.sum]], 'lats':[lats,[np.median]],
                        'longs':[longs,[np.median]]}}

avgs_feats = [all_avgs]


# In[ ]:

# A similar example using a surrounding grid for each station. There are no lat or long options for this type of
# feature set

# Dimensions without aggregation

# Use all variables
var = ['all']

# Keep model 0 (the default model) as a column for each of the variables (aggregated over other dimensions)
model1 = [0]

# Dimensions with aggregation

# Aggregate over all other models (excluding model 0, which is used directly)
model2 = range(1,11)

# Aggregate over all times
times = ['all']

# Create a column for each member of the grid. All or nothing for gefs now. Could specify but currently see no need
# for it. We could also take an aggregate measure of the gefs (including interpolate). That doesn't work for the
# other dimensions

gefs = ['all']

grid_avgs = {'type':'relative', 'full_axes':{'var':var, 'models':model1, 'gefs':gefs},
            'agg_axes':{'models':[model2,[np.mean]], 'times':[times, [np.mean, np.sum]]}}

grid_feats = [grid_avgs]


# In[31]:

# A similar example using a surrounding grid for each station. Now, just average over the grid

# Dimensions without aggregation

# Use all variables
var = ['all']

# Keep model 0 (the default model) as a column for each of the variables (aggregated over other dimensions)
model1 = [0]

# Dimensions with aggregation

# Aggregate over all other models (excluding model 0, which is used directly)
model2 = range(1,11)

# Aggregate over all times
times = ['all']

# Create a column for each member of the grid. All or nothing for gefs now. Could specify but currently see no need
# for it. We could also take an aggregate measure of the gefs (including interpolate). That doesn't work for the
# other dimensions

gefs = ['all']

grid_avgs = {'type':'relative', 'full_axes':{'var':var, 'models':model1},
            'agg_axes':{'models':[model2,[np.mean]], 'times':[times, [np.mean, np.sum]], 'gefs':[gefs, [np.mean]]}}

gefs_mean_feats = [grid_avgs]


# In[36]:

# This just uses the station_names as another feature
stat_names = {'type':'station_names'}
stat_feats = [stat_names]


# In[37]:

frac_dist = {'type':'frac_dist'}
dist_feats = [frac_dist]


# In[52]:

# if I am modifying code for any of these pythons
reload(solar.wrangle.wrangle)
reload(solar.wrangle.subset)
reload(solar.wrangle.engineer)
from solar.wrangle.wrangle import SolarData

dist_data = SolarData.load(Xtrain_dir, ytrain_file, Xtest_dir, station_file,                                   train_dates, test_dates, station,                                   station_layout, dist_feats)


# In[38]:

days_solstice = {'type':'days_from_solstice'}
days_cold = {'type':'days_from_coldest'}
days_feats = [days_solstice, days_cold]


# In[ ]:

# if I am modifying code for any of these pythons
reload(solar.wrangle.wrangle)
reload(solar.wrangle.subset)
reload(solar.wrangle.engineer)
from solar.wrangle.wrangle import SolarData

dist_data = SolarData.load(Xtrain_dir, ytrain_file, Xtest_dir, station_file,                                   train_dates, test_dates, station,                                   station_layout, days_feats)


# In[39]:

all_feats = [frac_dist, days_cold, days_solstice, default_grid, stat_names]


# In[62]:

# if I am modifying code for any of these pythons
reload(solar.wrangle.wrangle)
reload(solar.wrangle.subset)
reload(solar.wrangle.engineer)
from solar.wrangle.wrangle import SolarData

all_feats_data = SolarData.load(Xtrain_dir, ytrain_file, Xtest_dir, station_file,                                   train_dates, test_dates, station,                                   station_layout, all_feats)


# In[63]:

all_feats_data[0]


# In[64]:

reload(solar.analyze.model)
import numpy as np
from solar.analyze.model import Model

from sklearn.linear_model import Ridge
from sklearn import metrics

error_formula = 'mean_absolute_error'
cv_splits = 10

write = 'local'
njobs = 1

model = Model.model(all_feats_data, Ridge, {'alpha':np.logspace(-5,-2,8,base=10)}, cv_splits,
                    error_formula, njobs, write, normalize=True, random_state=1)


# In[67]:

reload(solar.analyze.model)
import numpy as np
from solar.analyze.model import Model

from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
cv_splits = 10

error_formula = 'mean_absolute_error'
njobs = 1
write = 'local'
get_ipython().magic(u"prun model = Model.model(all_feats_data, GradientBoostingRegressor, {'n_estimators':[300]},                     cv_splits, error_formula, njobs, write, loss='ls', max_depth=1, random_state=0, learning_rate=.001)")


# In[61]:

reload(solar.analyze.model)
import numpy as np
from solar.analyze.model import Model

from sklearn.linear_model import Ridge
from sklearn import metrics

error_formula = 'mean_absolute_error'
cv_splits = 10

model = Model.model_from_pickle('input_2016-02-06-15-38-35.p', Ridge, {'alpha':np.logspace(5,10,10,base=10)}, cv_splits,
                    error_formula, normalize=True, random_state=1)


# In[60]:

reload(solar.report.submission)
from solar.report.submission import Submission

preds = Submission.make_submission_file(model, all_feats_data[1], all_feats_data[2], True)


# In[63]:

reload(solar.report.submission)
from solar.report.submission import Submission

preds = Submission.make_submission_file(model, all_feats_data[1], all_feats_data[2], True)


# In[64]:

import solar.report.submission
reload(solar.report.submission)
from solar.report.submission import Submission
from solar.analyze.model import Model
import cPickle as pickle

model = pickle.load(open('solar/data/kaggle_solar/models/model_2016-02-21-16-11-01.p', 'rb'))


# In[66]:

model.best_estimator_.feature_importances_


# In[76]:

import numpy as np
coefs = model.best_estimator_.feature_importances_.ravel()
var_import = []
for col, val in enumerate(list(coefs)):
    var_import.append((abs(val), col))
var_import.sort(reverse=True)
for i, entry in enumerate(var_import):
    print i+1, entry


# In[84]:

my_input = pickle.load(open('solar/data/kaggle_solar/inputs/input_2016-02-21-09-27-29.p', 'rb'))


# In[81]:

[(((var[1]-4)/15)/4)/5 for var in var_import][0:20]


# In[99]:

my_input[0]


# In[63]:

my_input[0].iloc[0:10,4+8+15*4]


# In[ ]:




# In[74]:

import cPickle as pickle
model = pickle.load(open('./solar/data/kaggle_solar/models/model_2016-02-21-16-11-01.p','rb'))


# In[75]:

model.best_estimator_.feature_importances_


# In[ ]:




# In[108]:

from sklearn.metrics import mean_absolute_error


# In[109]:

mean_absolute_error(my_input[1], [16526096]*len(my_input[1]))


# In[107]:

np.sum((abs(my_input[1].iloc[0:98]-16526096)).values)


# In[110]:

averages = [16877461.91,16237533.98,17119188.79,17010565.32,17560172.95,17612143.11,17304074.16,15969634.24,16061706.56,
 18688943.33,16034080.62,16655128.52,16014657.87,17304512.26,16104884.18,16438717.4,17331768.39,16597616.48,17515474.77,16055064.62,
 15993253.01,17015639.69,17484171.18,16541992.87,15486166.08,15656934.05,15667279.87,15896897.03,16348532.74,
 16707117.53,17440381.55,15718914.29,16444023.44,16082322.25,17182922.64,16936510.22,17943529.54,16526024.35,16066904.41,
 16924262.95,17184734.64,17988183.75,18041482.55,16184172.02,15849509.66,15679748.78,18700559.48,16715044.14,16741635.54,15952446.49,
 16367520.26,17439983.73,16440133.58,16786662.48,15709795.78,16240623.78,17040788.48,15766297.46,16626777.31,14795920.57,16228119.75,
 16659807.74,15986616.42,16009687.23,15959177.06,15622161.7,16284096.52,16192237.83,16322837.71,15657444.98,16890165.69,16355147.03,
 17303377.43,17170127.35,15721493.12,16952355.28,16359889.91,15904617.52,17262330.16,16441963.35,15716436.06,16101732.84,15903216.55,
 16688078.17,15826361.38,15590790.71,17739361.48,15941223.45,15737219.14,16648738.63,16959415.48,17039770.7,17074429.12,15331710.15,
 16000551.34,15657185.09,17423402.11,16085012.47]


# In[112]:

mean_absolute_error(my_input[1], averages*182)


# In[113]:

lin_model = pickle.load(open('./solar/data/kaggle_solar/models/model_2016-02-22-11-01-56.p','rb'))


# In[116]:

mean_absolute_error(my_input[1], lin_model.predict(my_input[0]))


# In[117]:

ridge_model = pickle.load(open('./solar/data/kaggle_solar/models/model_2016-02-22-12-03-14.p','rb'))


# In[118]:

mean_absolute_error(my_input[1], ridge_model.predict(my_input[0]))


# In[119]:

dt_model = pickle.load(open('./solar/data/kaggle_solar/models/model_2016-02-22-12-11-36.p','rb'))


# In[120]:

dt_model


# In[123]:

mean_absolute_error(my_input[1], dt_model.predict(my_input[0]))


# In[124]:

dt_model.predict(my_input[0]).shape


# In[125]:

gbm_model = pickle.load(open('./solar/data/kaggle_solar/models/model_2016-02-22-12-23-26.p','rb'))


# In[126]:

mean_absolute_error(my_input[1], gbm_model.predict(my_input[0]))

