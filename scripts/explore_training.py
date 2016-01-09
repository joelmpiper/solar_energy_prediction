""" This file is used for basic data exploration. This is based on the
    initial starter code from Alec Randford posted in the Kaggle forum.

    Alec's thanks: Some of this code is recycled for Miroslaw and Paul Duan's
    forum code from the Amazon Challenge, thanks!
"""

import pandas as pd
import itertools
import datetime
import numpy as np
import os
import netCDF4 as nc
import cPickle as pickle


def create_index(first_file):
    """ Create the indices that will be used for all of the variables.
    """

    # get the values stored in the variables of the nc file
    values = first_file.variables.values()
    # some minor manipulations to get the models into a string
    # the times and dates into the python datetime format
    models = [str(val) for val in values[4][:]]
    times = [datetime.time(val%24) for val in values[5][:]]
    dates = [datetime.datetime.strptime(str(val), "%Y%m%d%H")
             for val in values[1][:]]
    # return every combination of those values; this should align
    # to each reading of the variables
    return list(itertools.product(dates, models, times, values[2][:],
                                  values[3][:]))


def load_GEFS_data(directory,files_to_use,file_sub_str):
    """ Loads a list of GEFS files merging them into model format.

        Stolen from the beat the benchmark code by Alec Randford
    """
    for i, f in enumerate(files_to_use):
        # for the first file, load the index too
        if i == 0:
            X = load_GEFS_file(directory,files_to_use[i],file_sub_str)
            input_index = np.asarray(create_index(X))
            # The variable of interest is always in the last column
            first_var = X.variables.values()[-1]
            # Change for the five-dimensional format to one-dimensional
            single_col = np.reshape(first_var,-1)
            print('Loading file: ' + f)
            print('Dimensions of variables: ')
            print single_col.shape
            print('Dimensions of index: ')
            print input_index.shape
            full_values = np.hstack((input_index, single_col[:,None]))
            print('Loaded file: ' + f)
            print(str(i+1) + ' files completed.')
        else:
            X_new = load_GEFS_file(directory,files_to_use[i],file_sub_str)
            next_var = X_new.variables.values()[-1]
            next_col = np.reshape(next_var,-1)
            print('Loading file: ' + f)
            full_values = np.hstack((full_values,next_col[:,None]))
            print('Loaded file: ' + f)
            print(str(i+1) + ' files completed.')
    return pd.DataFrame(full_values)


def load_GEFS_file(directory,data_type,file_sub_str):
    """ Loads GEFS file using specified merge technique.

        Stolen from beat the benchmark fro Alec Randford.
    """

    print 'loading',data_type
    path = os.path.join(directory,data_type+file_sub_str)
    X = nc.Dataset(path,'r+')

   return X


def create_all_input():
    """ Create one huge dataframe with all of the nc data.

        Use some reasonable defaults to lead the input data
        Create the pickle file that will can store all of the data
    """

    data_dir='../data/kaggle_solar/train/'
    files_to_use='all'
    if files_to_use == 'all':
        files_to_use = ['dswrf_sfc','dlwrf_sfc','uswrf_sfc','ulwrf_sfc',
                  'ulwrf_tatm','pwat_eatm','tcdc_eatm','apcp_sfc','pres_msl',
                  'spfh_2m','tcolc_eatm','tmax_2m','tmin_2m','tmp_2m','tmp_sfc']
    train_sub_str = '_latlon_subset_19940101_20071231.nc'
    test_sub_str = '_latlon_subset_20080101_20121130.nc'


    print 'Loading training data...'
    train_input = load_GEFS_data(data_dir,files_to_use,train_sub_str)
    pickle.dump(train_input, open("full_input.p","wb"))

    return train_input
