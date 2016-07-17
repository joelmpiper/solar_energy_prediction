""" This file returns specified subsets of all relevant data as an input array.
    This can reduce the file sizes associated with all input, training, and
    test sets.
    joelmpiper [at] gmail.com
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
# import pdb

TEST = 0


class Subset(object):
    """ This returns the appropriate complementary training X, y and testing
    X and y.
    """

    def __init__(self, trainX_file, trainy_file, testX_file, testy_file):
        self.trainX_file = trainX_file
        self.trainy_file = trainy_file
        self.testX_file = testX_file
        self.trainX, self.trainy, self.testX, self.loc = Subset.subset(
            self.trainX_file, self.trainy_file, self.testX_file,
            self.testy_file)

    @staticmethod
    def subset(trainX_dir, trainy_file, testX_dir, loc_file, X_par, y_par):
        """ Returns the X, y, and location data cut by the parameters specified.

        Take any list of dates, times, models, latitudes, longitudes,
        and predictive locations, as well as elevation ranges.
        """

        trainX, testX = Subset.create_Xdata(trainX_dir, testX_dir, **X_par)
        trainy, loc = Subset.create_ydata(trainy_file, loc_file, **y_par)
        return trainX, trainy, testX, loc

    @staticmethod
    def create_ydata(train_file, loc_file, train_dates, stations):
        """ Returns a dataframes of the y data and the location data.

            Create dataframes by reading in the training and location files;
            tidy the training set by moving the location code from a column to
            to a row.
        """
        train_data = pd.read_csv(train_file, index_col='Date',
                                 parse_dates=['Date'])
        rot_df = pd.DataFrame(train_data.unstack().reset_index()[:])
        rot_df.rename(columns={'level_0': 'location', 'Date': 'date',
                               0: 'total_solar'}, inplace=True)
        rot_df.set_index('date', inplace=True)
        loc_df = pd.read_csv(loc_file, index_col='stid')

        loc_df['elon'] = loc_df['elon'] + 360
        loc_df.rename(columns={'nlat': 'lat', 'elon': 'lon'}, inplace=True)
        loc_df.index.names = ['location']

        # if we are making any specifications on the output data, then
        # reduce what is returned to conform with those specifications
        rot_df, loc_df = Subset.reduced_training(
            rot_df, loc_df, train_dates, stations)
        return rot_df, loc_df

    @staticmethod
    def reduced_training(rot_df, loc_df, date, station):
        """ Returns the y output and location data subsets given the parameters.

            Take the output data and location data and return the reduced set.
        """

        # only return data for the given stations
        mod_train = rot_df[rot_df['location'].isin(station)]
        mod_loc = loc_df.loc[station, :]

        # only return data for the given dates
        mod_train = mod_train.loc[date[0]:date[1], :].reset_index()
        mod_train = mod_train.sort_values(['location', 'date'])
        mod_train = mod_train.set_index('date')

        return mod_train, mod_loc

    @staticmethod
    def create_Xdata(trainX_dir, testX_dir, var=[], train_dates=[],
                     test_dates=[], models=[], lats=[], longs=[],
                     times=[], elevs=[], station=[]):
        """ Return an np array of X data with the given cuts.
        """
        # The default is all of the files, so load them all if none are
        # specified
        if ((var.size == 0) or ('all' in var)):
            var = [
                'dswrf_sfc', 'dlwrf_sfc', 'uswrf_sfc', 'ulwrf_sfc',
                'ulwrf_tatm', 'pwat_eatm', 'tcdc_eatm', 'apcp_sfc', 'pres_msl',
                'spfh_2m', 'tcolc_eatm', 'tmax_2m', 'tmin_2m', 'tmp_2m',
                'tmp_sfc'
            ]

        train_sub_str = '_latlon_subset_19940101_20071231.nc'
        test_sub_str = '_latlon_subset_20080101_20121130.nc'
        trainX = Subset.file_loop(trainX_dir, var, train_sub_str,
                                  train_dates, models, lats, longs, times,
                                  elevs)
        testX = Subset.file_loop(testX_dir, var, test_sub_str,
                                 test_dates, models, lats, longs, times,
                                 elevs)
        return trainX, testX

    @staticmethod
    def file_loop(input_dir, var, substr, dates, models, lats, longs,
                  times, elevs):
        """ Return an np array for the given set of variables and cuts.
        """

        # First loop over the files for the specificed variables in the set
        # Save each variables in a list. Each element is an ndarray
        loaded = []

        for i, f in enumerate(var):
            loaded.append(Subset.cutX(
                input_dir + '/' + var[i] + substr, dates, models, lats,
                longs, times, elevs))

        # Combine all of the files into a single array
        X = np.stack((loaded), axis=-1)
        return X

    @staticmethod
    def cutX(file_name, input_date, models=None,
             lats=None, longs=None,
             times=None, elevs=None):
        """ Load a single nc file and return a subset of that file in an
        np array.
        """

        X = nc.Dataset(file_name, 'r+').variables.values()

        # set defaults
        begin_date = 0
        end_date = len(list(X)[1][:])
        # if date values are provided revise the ranges
        if (input_date):
            dates = list(X)[1][:]
            begin_date, end_date = Subset.check_dates(dates, input_date)
            # For some reason in python 3, these were coming back as
            # 1-element lists
            begin_date = begin_date[0]
            end_date = end_date[0]

        X = np.array(list(X)[-1])[begin_date:(end_date+1),
                                  models[:, np.newaxis, np.newaxis, np.newaxis],
                                  times[:, np.newaxis, np.newaxis], lats, longs]
        if (TEST > 0):
            print("Subset complete")

        return X

    @staticmethod
    def check_dates(dates, input_date):
        """ Take a list of dates, a beginning date, and an ending date
        and return the indices needed to extract information from the
        np array.
        """

        beg_split = input_date[0].split('-')
        beg_ind = int(beg_split[0] + beg_split[1] + beg_split[2] + '00')
        begin_date = np.where(dates == beg_ind)[0]

        end_split = input_date[1].split('-')
        end_ind = int(end_split[0] + end_split[1] + end_split[2] + '00')
        end_date = np.where(dates == end_ind)[0]

        return begin_date, end_date


if __name__ == '__main__':
    s = Subset(trainX_file='solar/data/kaggle_solar/train/',
               trainy_file='solar/data/kaggle_solar/train.csv',
               testX_file='solar/data/kaggle_solar/test/',
               loc_file='solar/data/kaggle_solar/station_info.csv')
    print(s.trainX.shape)
