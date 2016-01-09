""" This file returns specified subsets of all relevant data as dataframes.

    joelmpiper@gmail.com
"""

import pandas as pd
import itertools
import datetime
import numpy as np
import os
import netCDF4 as nc

def create_training_data(train_dir, station=[], lat=[], lon=[],
                         elev=[]):
    """ Returns a dataframes of the y data and the location data.

        Create dataframes by reading in the training and location files;
        tidy the training set by moving the location code from a column to
        to a row.
    """
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


def reduced_training(rot_df, loc_df, station, lat, lon, elev):
    """ Returns the y output and location data subsets given the parameters.

        Take the output data and location data and return the reduced set.
    """

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


def cut_var(train, loc, var, var_name, train_split, loc_split):
    """ Take output dataframes and reduce the single variable. """

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


def extract_subset(input_dir, output_dir, **kwargs):
    """ Returns the X, y, and location data cut by the parameters specified.

        Take any list of dates, times, models, latitudes, longitudes,
        and predictive locations, as well as elevation ranges.
    """

    train_params = {}
    for key, value in kwargs.iteritems():
        if key in ['station', 'lat', 'lon', 'elev']:
            train_params[key] = value
    rot_df, loc_df = create_training_data(input_dir, **kwargs)
    return rot_df, loc_df
