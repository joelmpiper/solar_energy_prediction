
# -*- coding: utf-8 -*-

""" This file inlcudes the SolarData class. Although it is a class, I am
currently not really using the object-oriented functionality and instead
am just using the class as a namespace for static functions. There is still
a lot of work to do to get this functioning appropriately.


This includes the beat-the-benchmark files that are associated with
the initial data wrangling. See the URL to the forum post with the code:
https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest/forums/t/5282/beating-the-benchmark-in-python-2230k-mae
by Alec Radford.



"""

import os
import netCDF4 as nc
import numpy as np
import cPickle as pickle
import pandas as pd
import itertools
from solar.wrangle.subset import Subset
from solar.wrangle.engineer import Engineer


class SolarData(object):
    """ Load the solar data from the gefs and mesonet files """

    def __init__(self, benchmark=False, pickle=False,
                 trainX_dir='solar/data/kaggle_solar/train/',
                 testX_dir='solar/data/kaggle_solar/test/',
                 meso_dir='solar/data/kaggle_solar/',
                 trainy_file='solar/data/kaggle_solar/train.csv',
                 loc_file='solar/data/kaggle_solar/station_info.csv',
                 files_to_use=['dswrf_sfc'],
                 learn_type='one_to_day',
                 **kwargs):

        self.benchmark = benchmark
        self.pickled = pickle
        if(benchmark):
            self.data = self.load_benchmark(trainX_dir, testX_dir, meso_dir,
                                            files_to_use)
        else:
            if(pickle):
                self.data = self.load_pickle()
            else:
                self.data = self.load(
                    trainX_dir, trainy_file, testX_dir, loc_file, learn_type,
                    **kwargs)

    def load_benchmark(self, gefs_train, gefs_test, meso_dir, files_to_use):
        """ Returns numpy arrays, of the training data """

        '''
        Loads a list of GEFS files merging them into model format.
        '''
        def load_GEFS_data(directory, files_to_use, file_sub_str):
            for i, f in enumerate(files_to_use):
                if i == 0:
                    X = load_GEFS_file(directory, files_to_use[i], file_sub_str)
                else:
                    X_new = load_GEFS_file(directory, files_to_use[i],
                                           file_sub_str)
                    X = np.hstack((X, X_new))
            return X

        '''
        Loads GEFS file using specified merge technique.
        '''
        def load_GEFS_file(directory, data_type, file_sub_str):
            path = os.path.join(directory, data_type+file_sub_str)
            X = nc.Dataset(path, 'r+').variables.values()[-1][:, :, :, :, :]
            X = np.mean(X, axis=1)
            X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
            return X

        '''
        Load csv test/train data splitting out times.
        '''
        def load_csv_data(path):
            data = np.loadtxt(path, delimiter=',', dtype=float, skiprows=1)
            times = data[:, 0].astype(int)
            Y = data[:, 1:]
            return times, Y

        if files_to_use == 'all':
            files_to_use = ['dswrf_sfc', 'dlwrf_sfc', 'uswrf_sfc', 'ulwrf_sfc',
                            'ulwrf_tatm', 'pwat_eatm', 'tcdc_eatm', 'apcp_sfc',
                            'pres_msl', 'spfh_2m', 'tcolc_eatm', 'tmax_2m',
                            'tmin_2m', 'tmp_2m', 'tmp_sfc']
        train_sub_str = '_latlon_subset_19940101_20071231.nc'
        test_sub_str = '_latlon_subset_20080101_20121130.nc'

        print 'Loading training data...'
        trainX = load_GEFS_data(gefs_train, files_to_use, train_sub_str)
        times, trainY = load_csv_data(os.path.join(meso_dir, 'train.csv'))
        print 'Training data shape', trainX.shape, trainY.shape

        testX = load_GEFS_data(gefs_test, files_to_use, test_sub_str)
        pickle.dump((trainX, trainY, times, testX),
                    open("solar/data/kaggle_solar/all_input.p", "wb"))
        return trainX, trainY, times, testX

    @staticmethod
    def load_data_explore(trainX_dir, trainy_file, testX_dir, loc_file,
                          X_par, y_par, station, eng_feat):
        if (eng_feat):
            train_X, train_y, test_X, loc = Engineer.engineer(
                trainX_dir, trainy_file, testX_dir, loc_file,
                X_par, y_par, eng_feat)
        else:
            train_X, train_y, test_X, loc = Subset.subset(
                trainX_dir, trainy_file, testX_dir, loc_file, X_par, y_par)

        # Reshape into one column for exploration
        train_X = train_X.reshape(np.product(train_X.shape[:]))
        test_X = test_X.reshape(np.product(test_X.shape[:]))
        train_index, test_index = SolarData.create_index(
            eng_feat, station, **X_par)
        trainX_df = pd.merge(train_index, pd.DataFrame(train_X),
                             left_index=True, right_index=True)
        testX_df = pd.merge(test_index, pd.DataFrame(test_X),
                            left_index=True, right_index=True)

        # Reorder so that these can be used like variables
        if (eng_feat):
            trainX_df = trainX_df.set_index(
                ['variable', 'date', 'model', 'time', 'station',
                 'lat_longs']).unstack('variable')[0]
            testX_df = testX_df.set_index(
                ['variable', 'date', 'model', 'time', 'station',
                 'lat_longs']).unstack('variable')[0]
        else:
            trainX_df = trainX_df.set_index(['variable', 'date', 'model',
                                             'time', 'lat', 'lon']).unstack(
                                                 'variable')[0]
            testX_df = testX_df.set_index(['variable', 'date', 'model',
                                           'time', 'lat', 'lon']).unstack(
                                               'variable')[0]

        # although these can be thought of as indices, they are easier to
        # manipulate as regular columns
        trainX_df.reset_index(inplace=True)
        testX_df.reset_index(inplace=True)

        return trainX_df, train_y, testX_df, loc

    @staticmethod
    def create_index(eng_feat, station_names, train_dates=[], test_dates=[],
                     var=[], models=[], lats=[], longs=[], times=[],
                     station=[]):
        """ Create an index based on the input X parameters and the
        engineered features.
        """

        if (train_dates):
            tr_date_index = np.arange(
                train_dates[0], np.datetime64(train_dates[1])+1,
                dtype='datetime64')
        else:
            tr_date_index = np.arange(np.datetime64('1994-01-01'),
                                      np.datetime64('2008-01-01'),
                                      dtype='datetime64')
        if (test_dates):
            ts_date_index = np.arange(
                test_dates[0], np.datetime64(test_dates[1])+1,
                dtype='datetime64')
        else:
            ts_date_index = np.arange(np.datetime64('2008-01-01'),
                                      np.datetime64('2012-12-01'),
                                      dtype='datetime64')
        if ((var.size == 0) or ('all' in var)):
            var = [
                'dswrf_sfc', 'dlwrf_sfc', 'uswrf_sfc', 'ulwrf_sfc',
                'ulwrf_tatm', 'pwat_eatm', 'tcdc_eatm', 'apcp_sfc', 'pres_msl',
                'spfh_2m', 'tcolc_eatm', 'tmax_2m', 'tmin_2m', 'tmp_2m',
                'tmp_sfc']

        # if we are using a grid, ignore lat and long and just use
        # how the points are relative to station
        time_ind = [time*3 + 12 for time in times]
        model_ind = [model for model in models]
        if ('grid' in eng_feat):
            lat_longs = ['SE', 'SW', 'NE', 'NW']
            col_names = ['date', 'model', 'time', 'station',
                         'lat_longs', 'variable']
            new_tr_indices = pd.DataFrame(list(itertools.product(
                tr_date_index, model_ind, time_ind, station_names,
                lat_longs, var)))
            new_tr_indices.columns = col_names
            new_ts_indices = pd.DataFrame(list(itertools.product(
                ts_date_index, model_ind, time_ind, station_names,
                lat_longs, var)))
            new_ts_indices.columns = col_names

        else:
            col_names = ['variable', 'date', 'model', 'time', 'lat', 'lon']
            lat_ind = lats[0] + 34
            lon_ind = longs + 254
            new_tr_indices = pd.DataFrame(list(itertools.product(
                var, tr_date_index, model_ind, time_ind, lat_ind, lon_ind)))
            new_tr_indices.columns = col_names

            new_ts_indices = pd.DataFrame(list(itertools.product(
                var, ts_date_index, model_ind, time_ind, lat_ind, lon_ind)))
            new_ts_indices.columns = col_names

        return new_tr_indices, new_ts_indices

    @staticmethod
    def load_data_learn(trainX_dir, trainy_file, testX_dir, loc_file,
                        X_par, y_par, station, eng_feat):
        """ Create data that can fed into a learning model
        """

        train_X, train_y, test_X, __ = SolarData.load_data_explore(
            trainX_dir, trainy_file, testX_dir, loc_file, X_par, y_par,
            station, eng_feat)
        if (eng_feat):
            train_X = train_X.reset_index().set_index(
                ['date', 'model', 'time', 'station', 'lat_longs']).unstack(
                    ['model', 'time', 'lat_longs'])
            test_X = test_X.reset_index().set_index(
                ['date', 'model', 'time', 'station', 'lat_longs']).unstack(
                    ['model', 'time', 'lat_longs'])
            train_y = train_y.reset_index().set_index(
                ['date', 'location']).sort_index()
        else:
            train_X = train_X.reset_index().set_index(
                ['date', 'model', 'time', 'lat', 'lon']).unstack(
                    ['model', 'time', 'lat', 'lon'])
            test_X = test_X.reset_index().set_index(
                ['date', 'model', 'time', 'lat', 'lon']).unstack(
                    ['model', 'time', 'lat', 'lon'])
            train_y.reset_index(inplace=True)
            train_y = train_y.pivot(index='date', columns='location',
                                    values='total_solar')

        train_X.drop('index', axis=1, inplace=True)
        test_X.drop('index', axis=1, inplace=True)

        return train_X, train_y, test_X

    @staticmethod
    def load_trainy(yfile, locfile, dates, stations, stat_layout):
        """ Return the y values for the training data, given the file
        location, the location file, the training dates, the stations,
        and the final layout.
        """

        mod_stations = SolarData.parse_parameters(station=stations)
        trainy, locy = Subset.create_ydata(yfile, locfile, dates,
                                           mod_stations['station'])
        trainy = trainy.reset_index()
        if (stat_layout):
            trainy = trainy.set_index(['date', 'location']).sort_index()
        else:
            trainy = trainy.pivot(index='date', columns='location',
                                  values='total_solar')
        return trainy, locy

    @staticmethod
    def make_features(trainX_dir, testX_dir, locy, train_dates, test_dates,
                      stations, station_layout, features):
        """ Take relevant directories, features, and other parameters, and
        return testing and training dataframes.
        """

        # iterate over each of the included features and return the relevant
        # parameters
        for feat_type in features:
            if (feat_type['type'] == 'relative'):

                X_par = SolarData.parse_parameters(**features[0]['axes'])
                X_par['train_dates'] = train_dates
                X_par['test_dates'] = test_dates

                eng_feat = 'grid'
                trainX, testX = Engineer.engineer(
                    trainX_dir, testX_dir,
                    locy, X_par,
                    train_dates, stations, eng_feat)

        # Reshape into one column for exploration
        trainX = trainX.reshape(np.product(trainX.shape[:]))
        testX = testX.reshape(np.product(testX.shape[:]))
        stations = X_par['station']
        del X_par['station']
        train_index, test_index = SolarData.create_index(
            eng_feat, stations, **X_par)
        trainX = pd.merge(train_index, pd.DataFrame(trainX),
                          left_index=True, right_index=True)
        testX = pd.merge(test_index, pd.DataFrame(testX),
                         left_index=True, right_index=True)
        # manipulated no matter the form that they returned in

        # start by resetting the index so that these can be more easily
        trainX = trainX.reset_index()
        testX = testX.reset_index()

        # the included columns will be different depending on the included
        # features

        trainX.rename(columns={0: features[0]['type']}, inplace=True)
        testX.rename(columns={0: features[0]['type']}, inplace=True)
        trainX.drop('index', axis=1, inplace=True)
        testX.drop('index', axis=1, inplace=True)
        all_cols = list(trainX.columns)[:-1]

        # we don't want the indices to be columns, so use all of the columns
        # for indices
        trainX = trainX.set_index(all_cols)
        testX = testX.set_index(all_cols)

        # leave the row indices depending on what the data should look like
        # for the training model; either one row per date, or one row per
        # date and station
        diff_cols = []
        if (station_layout):
            diff_cols = {'date', 'station'}
        else:
            diff_cols = {'date'}

        # With the known columns move the other indices to the top
        trainX = trainX.unstack(list(set(all_cols).difference(diff_cols)))
        testX = testX.unstack(list(set(all_cols).difference(diff_cols)))

        return trainX, testX

    @staticmethod
    def load(trainX_dir, trainy_file, testX_dir, loc_file, train_dates,
             test_dates, stations, station_layout, features):
        """ Returns numpy arrays, of the training data """

        trainy, locy = SolarData.load_trainy(
            trainy_file, loc_file, train_dates, stations, station_layout)

        trainX, testX = SolarData.make_features(
            trainX_dir, testX_dir, locy, train_dates, test_dates, stations,
            station_layout, features)

        pickle.dump((trainX, trainy, testX, locy),
                    open("solar/data/kaggle_solar/all_input.p", "wb"))
        return trainX, trainy, testX, locy

    @staticmethod
    def fill_default(param):
        if (param == 'station'):
            return [
                'ACME', 'ADAX', 'ALTU', 'APAC', 'ARNE', 'BEAV', 'BESS',
                'BIXB', 'BLAC', 'BOIS', 'BOWL', 'BREC', 'BRIS', 'BUFF',
                'BURB', 'BURN', 'BUTL', 'BYAR', 'CAMA', 'CENT', 'CHAN',
                'CHER', 'CHEY', 'CHIC', 'CLAY', 'CLOU', 'COOK', 'COPA',
                'DURA', 'ELRE', 'ERIC', 'EUFA', 'FAIR', 'FORA', 'FREE',
                'FTCB', 'GOOD', 'GUTH', 'HASK', 'HINT', 'HOBA', 'HOLL',
                'HOOK', 'HUGO', 'IDAB', 'JAYX', 'KENT', 'KETC', 'LAHO',
                'LANE', 'MADI', 'MANG', 'MARE', 'MAYR', 'MCAL', 'MEDF',
                'MEDI', 'MIAM', 'MINC', 'MTHE', 'NEWK', 'NINN', 'NOWA',
                'OILT', 'OKEM', 'OKMU', 'PAUL', 'PAWN', 'PERK', 'PRYO',
                'PUTN', 'REDR', 'RETR', 'RING', 'SALL', 'SEIL', 'SHAW',
                'SKIA', 'SLAP', 'SPEN', 'STIG', 'STIL', 'STUA', 'SULP',
                'TAHL', 'TALI', 'TIPT', 'TISH', 'VINI', 'WASH', 'WATO',
                'WAUR', 'WEAT', 'WEST', 'WILB', 'WIST', 'WOOD', 'WYNO']
        elif (param == 'models'):
            return np.arange(0, 11)
        elif (param == 'times'):
            return np.arange(0, 6)
        elif (param == 'lats'):
            return np.arange(0, 9)
        elif (param == 'longs'):
            return np.arange(0, 16)
        else:
            return None

    @staticmethod
    def make_index(key, val):
        value = np.array(val)
        if (key == 'times'):
            return (value - 12)//3
        elif (key == 'lats'):
            return value - 34
        elif (key == 'longs'):
            return value - 254
        else:
            return value

    @staticmethod
    def parse_parameters(**kwargs):
        """ Take a set of parameters and fill it with defaults. Return the
        passed parameters.
        """

        # Start building the return dictionary
        return_args = {}

        # iterate over all of the passed items
        for key, value in kwargs.iteritems():
            if ('all' in value):
                return_args[key] = SolarData.fill_default(key)
            else:
                return_args[key] = SolarData.make_index(key, value)

#        return_args['models']= (np.array(return_args['models']))[:, np.newaxis,
        #                                                          np.newaxis,
        #                                                          np.newaxis]

        return return_args

    def load_pickle(self, pickle_dir='solar/data/kaggle_solar/'):
        return pickle.load(open(pickle_dir + '/all_input.p', 'rb'))

if __name__ == '__main__':
    train_dates = ['2005-07-29', '2005-08-02']
    test_dates = ['2008-12-29', '2009-01-05']
    var = ['dswrf_sfc', 'uswrf_sfc', 'spfh_2m', 'ulwrf_tatm']
    lats = range(34, 37)
    longs = range(260, 264)
    models = [0]
    solar_array = SolarData(pickle=False, benchmark=False, var=var,
                            models=models, lats=lats, longs=longs,
                            train_dates=train_dates, test_dates=test_dates)
    print solar_array.data
