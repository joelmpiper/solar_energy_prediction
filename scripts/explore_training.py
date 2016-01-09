''' This is mostly to load the training data into a dataframe for easier
    exploration.

    File loading and reading copied from the starter code. See the intro for
    that below.

    This came from the modified starter code from Alec Randford, with his
    thanks below.

    Alec's thanks: Some of this code is recycled for Miroslaw and Paul Duan's
    forum code from the Amazon Challenge, thanks!
'''

import pandas as pd
import itertools
import datetime
import numpy as np
import os
import netCDF4 as nc
import cPickle as pickle

''' Create dataframes by reading in the training and location files;
    tidy the training set by moving the location code from a column to
    to a row
'''
def create_training_data(train_dir, station=[], lat=[], lon=[],
                         elev=[]):
    train_data = pd.read_csv(train_dir + '/train.csv',
                             index_col='Date', parse_dates=['Date'])
    rot_df = pd.DataFrame(train_data.unstack().reset_index()[:])
    rot_df.rename(columns={'level_0':'location','Date':'date',
                           0:'total_solar'}, inplace=True)
    rot_df.set_index('date',inplace=True)
    loc_df = pd.read_csv(train_dir + '/station_info.csv',
                           index_col='stid')

    loc_df['elon'] = loc_df['elon'] + 360
    loc_df.rename(columns={'nlat':'lat','elon':'lon'}, inplace=True)
    loc_df.index.names = ['station']

    # if we are making any specifications on the output data, then
    # reduce what is returned to conform with those specifications
    if(station or lat or lon or elev):
        rot_df, loc_df = reduced_training(rot_df, loc_df, station, lat,
                                         lon, elev)
    return rot_df, loc_df


''' Take the output data and location data and return the reduced set
'''
def reduced_training(rot_df, loc_df, station, lat, lon, elev):
    if (station):
        mod_train = rot_df[rot_df['location'].isin(station)]
        mod_loc = loc_df.loc[station,:]
    else:
        mod_train = rot_df
        mod_loc = loc_df
        if (lat):
            mod_train, mod_loc = cut_var(mod_train, mod_loc, lat, 'lat',
                    rot_df.columns, loc_df.columns)
        if (lon):
            mod_train, mod_loc = cut_var(mod_train, mod_loc, lon, 'lon',
                    rot_df.columns, loc_df.columns)
        if (elev):
            mod_train, mod_loc = cut_var(mod_train, mod_loc, elev, 'elev',
                    rot_df.columns, loc_df.columns)
    return mod_train, mod_loc


''' Reduce on a single variable '''
def cut_var(train, loc, var, var_name, train_split, loc_split):
    combined = train.merge(loc, left_on='location',
                                right_index=True)

    combined = combined[(combined[var_name] >= var[0]) &
                        (combined[var_name] <= var[1])]
    mod_train = combined[train_split].drop_duplicates()
    mod_loc = combined[loc_split.union(['location'])]
    mod_loc = mod_loc.reset_index().drop('date', axis=1)
    mod_loc.rename(columns={'location':'station'}, inplace=True)
    mod_loc = mod_loc.drop_duplicates().set_index('station')

    return mod_train, mod_loc


''' Create the indices that will be used for all of the variables.
'''
def create_index(first_file):
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


''' Stolen from the beat the benchmark code by
    Loads a list of GEFS files merging them into model format.
'''
def load_GEFS_data(directory,files_to_use,file_sub_str):
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


'''
    Stolen completely from beat-the-benchmark although I guess I get rid
    of most of the functionality for exploration.
    Loads GEFS file using specified merge technique.
'''
def load_GEFS_file(directory,data_type,file_sub_str):
    print 'loading',data_type
    path = os.path.join(directory,data_type+file_sub_str)
    X = nc.Dataset(path,'r+')
    return X


''' Use some reasonable defaults to lead the input data
    Create the pickle file that will can store all of the data
'''
def create_all_input():

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


''' Extract a subset of the data based on a set of lists that bounds
    the relevant variables.

def input_subset(file_list, dates, models, times, lats, lons):

    latitudes = d.variables['lat'][:]
    lat_i = np.where( latitudes == lat )[0][0]

    longitudes = d.variables['lon'][:]
    lon_i = np.where( longitudes == lon )[0][0]

    latitudes = map( int, list( latitudes ))
    longitudes = map( int, list( longitudes ))

    return sub_df
'''


''' Take any list of dates, times, models, latitudes, longitudes,
    and predictive locations, as well as elevation ranges.
'''
def extract_subset(input_dir, output_dir, **kwargs):

    train_params = {}
    for key, value in kwargs.iteritems():
        if key in ['station', 'lat', 'lon', 'elev']:
            train_params[key] = value
    rot_df, loc_df = create_training_data(input_dir, **kwargs)
    return rot_df, loc_df
