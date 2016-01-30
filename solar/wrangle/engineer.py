""" This file returns engineered features for all of the individual points on
specific days.

    joelmpiper [at] gmail.com
"""

import numpy as np
import pandas as pd
import itertools
from solar.wrangle.subset import Subset

TEST = 0


class Engineer(object):
    """ This returns a y created from the subset file and an X file created
    by engineering features for each individual data point.
    """

    def __init__(self, trainX_file):
        self.trainX_file = trainX_file

    @staticmethod
    def create_grid(trainX_dir, testX_dir, loc_df, train_dates, test_dates,
                    stations, feature):
        """ take the station north, east, south, and west of each of
        the given points, as well as any other specified cuts and
        return the new file.
        """

        Xparams = Engineer.parse_parameters(**feature['axes'])
        Xparams['train_dates'] = train_dates
        Xparams['test_dates'] = test_dates
        Xparams['station'] = stations

        Xparams['lats'] = []
        Xparams['longs'] = []
        for sta in loc_df.index:
            stlat = loc_df.loc[sta].lat
            stlong = loc_df.loc[sta].lon
            # Subtract 34 because we are storing indices
            lats = 2*[int(np.floor(stlat)) - 34] + 2*[
                int(np.ceil(stlat)) - 34]
            # Subtract 254 because we are storing indices
            longs = 2*[int(np.floor(stlong)) - 254, int(np.ceil(stlong)) - 254]
            Xparams['lats'].append(lats)
            Xparams['longs'].append(longs)

        return Subset.create_Xdata(trainX_dir, testX_dir, **Xparams)

    @staticmethod
    def engineer(trainX_dir, testX_dir, locy, train_dates, test_dates,
                 stations, feature):
        """ Add the passed feature and return the train and test X data
        and the indices that correspond to the data.
        """

        if (feature['type'] == 'relative'):
            trainX, testX = Engineer.make_relative(trainX_dir, testX_dir, locy,
                                                   train_dates, test_dates,
                                                   stations, feature)
        else:
            trainX, testX = None, None

        return trainX, testX

    @staticmethod
    def make_relative(trainX_dir, testX_dir, locy, train_dates, test_dates,
                      stations, feature):
        """ Create the features given the list of input variables, models,
        etc., for a grid of points surrounding the station list.
        """

        trainX, testX = Engineer.create_grid(trainX_dir, testX_dir, locy,
                                             train_dates, test_dates,
                                             stations, feature)

        # Reshape into one column for exploration
        trainX = trainX.reshape(np.product(trainX.shape[:]))
        testX = testX.reshape(np.product(testX.shape[:]))

        X_par = Engineer.parse_parameters(**feature['axes'])
        X_par['train_dates'] = train_dates
        X_par['test_dates'] = test_dates

        stations = X_par['station']
        del X_par['station']

        train_index, test_index = Engineer.create_index(
            feature['type'], stations, **X_par)
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

        trainX.rename(columns={0: feature['type']}, inplace=True)
        testX.rename(columns={0: feature['type']}, inplace=True)
        trainX.drop('index', axis=1, inplace=True)
        testX.drop('index', axis=1, inplace=True)

        return trainX, testX

    @staticmethod
    def fill_default(param):
        if (param == 'station'):
            return np.array((
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
                'WAUR', 'WEAT', 'WEST', 'WILB', 'WIST', 'WOOD', 'WYNO'))
        elif (param == 'models'):
            return np.arange(0, 11)
        elif (param == 'times'):
            return np.arange(0, 6)
        elif (param == 'lats'):
            return np.arange(0, 9)
        elif (param == 'longs'):
            return np.arange(0, 16)
        elif (param == 'var'):
            return np.array((
                'dswrf_sfc', 'dlwrf_sfc', 'uswrf_sfc', 'ulwrf_sfc',
                'ulwrf_tatm', 'pwat_eatm', 'tcdc_eatm', 'apcp_sfc', 'pres_msl',
                'spfh_2m', 'tcolc_eatm', 'tmax_2m', 'tmin_2m', 'tmp_2m',
                'tmp_sfc'))
        else:
            return None

    @staticmethod
    def make_index_num(key, val):
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
                return_args[key] = Engineer.fill_default(key)
            else:
                return_args[key] = Engineer.make_index_num(key, value)

        return return_args

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
        if ('relative' in eng_feat):
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
