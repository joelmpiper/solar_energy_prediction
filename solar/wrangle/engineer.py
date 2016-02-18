""" This file returns engineered features for all of the individual points on
specific days.

    joelmpiper [at] gmail.com
"""

import numpy as np
import pandas as pd
import itertools
import datetime

from solar.wrangle.subset import Subset

TEST = 0


class Engineer(object):
    """ This returns a y created from the subset file and an X file created
    by engineering features for each individual data point.
    """

    def __init__(self, trainX_file):
        self.trainX_file = trainX_file

    @staticmethod
    def frac_dist(locy):
        """ Return the fractional part of the latitude and longitude measures.
        """

        lat_dist = []
        long_dist = []
        for sta in locy.index:
            lat_dist.append(locy.loc[sta].lat % 1)
            long_dist.append(locy.loc[sta].lon % 1)
        return lat_dist, long_dist

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
            lats = 2*[int(np.floor(stlat)) - 31] + 2*[
                int(np.ceil(stlat)) - 31]
            # Subtract 254 because we are storing indices
            longs = 2*[int(np.floor(stlong)) - 254, int(np.ceil(stlong)) - 254]
            Xparams['lats'].append(lats)
            Xparams['longs'].append(longs)

        return Subset.create_Xdata(trainX_dir, testX_dir, **Xparams)

    @staticmethod
    def engineer(trainX_dir, testX_dir, locy, train_dates, test_dates,
                 stations, station_layout, feature):
        """ Add the passed feature and return the train and test X data
        and the indices that correspond to the data.
        """

        if (feature['type'] == 'relative'):
            trainX, testX = Engineer.make_relative(trainX_dir, testX_dir, locy,
                                                   train_dates, test_dates,
                                                   stations, station_layout,
                                                   feature)
        elif (feature['type'] == 'station_names'):
            trainX, testX = Engineer.make_station_names(
                train_dates, test_dates, stations, station_layout, feature)

        elif (feature['type'] == 'frac_dist'):
            trainX, testX = Engineer.make_frac_dist(
                locy, train_dates, test_dates, stations, station_layout,
                feature)

        elif (feature['type'] == 'days_from_solstice'):
            trainX, testX = Engineer.make_days_from_solstice(
                train_dates, test_dates, stations, station_layout, feature)

        elif (feature['type'] == 'days_from_coldest'):
            trainX, testX = Engineer.make_days_from_coldest(
                train_dates, test_dates, stations, station_layout, feature)

        else:
            trainX, testX = None, None

        return trainX, testX

    @staticmethod
    def make_frac_dist(locy, train_dates, test_dates, stations, station_layout,
                       feature):

        lat_dist, long_dist = Engineer.frac_dist(locy)
        ret = Engineer.fill_default(train_dates=train_dates,
                                    test_dates=test_dates,
                                    station=stations)

        stations = ret['station']

        test_len = len(ret['test_dates'])
        train_len = len(ret['train_dates'])

        trainX = [val for val in lat_dist for i in range(train_len)] +\
            [val for val in long_dist for i in range(train_len)]

        testX = [val for val in lat_dist for i in range(test_len)] +\
            [val for val in long_dist for i in range(test_len)]

        dist_ind = ['lat_dist', 'long_dist']

        train_index = Engineer.create_index(
            train_dates=train_dates, station=stations, dist_ind=dist_ind)
        test_index = Engineer.create_index(
            test_dates=test_dates, station=stations, dist_ind=dist_ind)

        trainX = pd.merge(train_index, pd.DataFrame(trainX),
                          left_index=True, right_index=True)
        testX = pd.merge(test_index, pd.DataFrame(testX),
                         left_index=True, right_index=True)

        # start by resetting the index so that these can be more easily
        trainX = trainX.reset_index()
        testX = testX.reset_index()

        # the included columns will be different depending on the included
        # features

        trainX.rename(columns={0: feature['type']}, inplace=True)
        testX.rename(columns={0: feature['type']}, inplace=True)
        trainX.drop('index', axis=1, inplace=True)
        testX.drop('index', axis=1, inplace=True)

        train_cols = list(trainX.columns)[:-1]
        test_cols = list(testX.columns)[:-1]

        # we don't want the indices to be columns, so use all of the columns
        # for indices
        trainX = trainX.set_index(train_cols)
        testX = testX.set_index(test_cols)
        trainX = trainX.swaplevel(0, 2, axis=0)
        testX = testX.swaplevel(0, 2, axis=0)

        if (station_layout):
            train_diff_cols = {'train_dates', 'station'}
            test_diff_cols = {'test_dates', 'station'}
        else:
            train_diff_cols = {'train_dates'}
            test_diff_cols = {'test_dates'}

        # With the known columns move the other indices to the top
        trainX = trainX.unstack(list(
            set(train_cols).difference(train_diff_cols)))
        testX = testX.unstack(list(
            set(test_cols).difference(test_diff_cols)))

        trainX = trainX.sort_index()
        testX = testX.sort_index()

        return trainX, testX

    @staticmethod
    def make_relative(trainX_dir, testX_dir, locy, train_dates, test_dates,
                      stations, station_layout, feature):
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

        lat_longs = ['SE', 'SW', 'NE', 'NW']
        train_index = Engineer.create_index(
            train_dates=train_dates, lat_longs=lat_longs, **feature['axes'])
        test_index = Engineer.create_index(
            test_dates=test_dates, lat_longs=lat_longs, **feature['axes'])
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

        train_cols = list(trainX.columns)[:-1]
        test_cols = list(testX.columns)[:-1]

        # we don't want the indices to be columns, so use all of the columns
        # for indices
        trainX = trainX.set_index(train_cols)
        testX = testX.set_index(test_cols)

        if (station_layout):
            train_diff_cols = {'train_dates', 'station'}
            test_diff_cols = {'test_dates', 'station'}
        else:
            train_diff_cols = {'train_dates'}
            test_diff_cols = {'test_dates'}

        # With the known columns move the other indices to the top
        trainX = trainX.unstack(list(
            set(train_cols).difference(train_diff_cols)))
        testX = testX.unstack(list(
            set(test_cols).difference(test_diff_cols)))

        trainX = trainX.sort_index()
        testX = testX.sort_index()

        return trainX, testX

    @staticmethod
    def list_days_from(input_list, compare_date):
        """ Return a list of the days away from a comparison date. """

        return [Engineer.days_from(date1, compare_date) for date1 in
                pd.DatetimeIndex(input_list)]

    @staticmethod
    def days_from(input_date, compare_date):
        """ Calculate the days between the two dates
        """

        con_comp = datetime.datetime.strptime(compare_date, '%Y-%m-%d')

        con_full = input_date

        comp_date = datetime.date(con_full.year, con_comp.month, con_comp.day)

        end_year = datetime.date(con_full.year, 12, 31)

        abs_diff = abs((con_full.timetuple().tm_yday) -
                       (comp_date.timetuple().tm_yday))

        if (abs_diff >= 183):
            final = end_year.timetuple().tm_yday - abs_diff
        else:
            final = abs_diff
        return final

    @staticmethod
    def make_days_from_coldest(train_dates, test_dates, stations,
                               station_layout, feature):
        """ Return how far the date is from Jan 4 (a coldest approximation).
        """

        ret = Engineer.fill_default(train_dates=train_dates,
                                    test_dates=test_dates,
                                    station=stations)

        train_cold = Engineer.list_days_from(ret['train_dates'],
                                             '1995-01-04')
        test_cold = Engineer.list_days_from(ret['test_dates'],
                                            '1995-01-04')

        test_len = len(ret['station'])
        train_len = len(ret['station'])

        trainX = train_cold * train_len
        testX = test_cold * test_len

        cold_ind = ['from_coldest']

        train_index = Engineer.create_index(
            train_dates=train_dates, station=stations, cold_ind=cold_ind)
        test_index = Engineer.create_index(
            test_dates=test_dates, station=stations, cold_ind=cold_ind)

        trainX = pd.merge(train_index, pd.DataFrame(trainX),
                          left_index=True, right_index=True)
        testX = pd.merge(test_index, pd.DataFrame(testX),
                         left_index=True, right_index=True)

        # start by resetting the index so that these can be more easily
        trainX = trainX.reset_index()
        testX = testX.reset_index()

        # the included columns will be different depending on the included
        # features

        trainX.rename(columns={0: feature['type']}, inplace=True)
        testX.rename(columns={0: feature['type']}, inplace=True)
        trainX.drop('index', axis=1, inplace=True)
        testX.drop('index', axis=1, inplace=True)

        train_cols = list(trainX.columns)[:-1]
        test_cols = list(testX.columns)[:-1]

        # we don't want the indices to be columns, so use all of the columns
        # for indices
        trainX = trainX.set_index(train_cols)
        testX = testX.set_index(test_cols)

        trainX = trainX.swaplevel(0, 2, axis=0)
        testX = testX.swaplevel(0, 2, axis=0)

        if (station_layout):
            train_diff_cols = {'train_dates', 'station'}
            test_diff_cols = {'test_dates', 'station'}
        else:
            train_diff_cols = {'train_dates'}
            test_diff_cols = {'test_dates'}

        # With the known columns move the other indices to the top
        trainX = trainX.unstack(list(
            set(train_cols).difference(train_diff_cols)))
        testX = testX.unstack(list(
            set(test_cols).difference(test_diff_cols)))

        trainX = trainX.sort_index()
        testX = testX.sort_index()

        return trainX, testX

    @staticmethod
    def make_days_from_solstice(train_dates, test_dates, stations,
                                station_layout, feature):
        """ Return how far the date is from Dec 21 (a solstice approximation).
        """

        ret = Engineer.fill_default(train_dates=train_dates,
                                    test_dates=test_dates,
                                    station=stations)

        train_solstice = Engineer.list_days_from(ret['train_dates'],
                                                 '1995-12-21')
        test_solstice = Engineer.list_days_from(ret['test_dates'],
                                                '1995-12-21')

        test_len = len(ret['station'])
        train_len = len(ret['station'])

        trainX = train_solstice * train_len
        testX = test_solstice * test_len

        sol_ind = ['from_solstice']

        train_index = Engineer.create_index(
            train_dates=train_dates, station=stations, sol_ind=sol_ind)
        test_index = Engineer.create_index(
            test_dates=test_dates, station=stations, sol_ind=sol_ind)

        trainX = pd.merge(train_index, pd.DataFrame(trainX),
                          left_index=True, right_index=True)
        testX = pd.merge(test_index, pd.DataFrame(testX),
                         left_index=True, right_index=True)

        # start by resetting the index so that these can be more easily
        trainX = trainX.reset_index()
        testX = testX.reset_index()

        # the included columns will be different depending on the included
        # features

        trainX.rename(columns={0: feature['type']}, inplace=True)
        testX.rename(columns={0: feature['type']}, inplace=True)
        trainX.drop('index', axis=1, inplace=True)
        testX.drop('index', axis=1, inplace=True)

        train_cols = list(trainX.columns)[:-1]
        test_cols = list(testX.columns)[:-1]
        # we don't want the indices to be columns, so use all of the columns
        # for indices
        trainX = trainX.set_index(train_cols)
        testX = testX.set_index(test_cols)
        trainX = trainX.swaplevel(0, 1, axis=0)
        testX = testX.swaplevel(0, 1, axis=0)

        if (station_layout):
            train_diff_cols = {'train_dates', 'station'}
            test_diff_cols = {'test_dates', 'station'}
        else:
            train_diff_cols = {'train_dates'}
            test_diff_cols = {'test_dates'}

        # With the known columns move the other indices to the top
        trainX = trainX.unstack(list(
            set(train_cols).difference(train_diff_cols)))
        testX = testX.unstack(list(
            set(test_cols).difference(test_diff_cols)))

        trainX = trainX.sort_index()
        testX = testX.sort_index()

        return trainX, testX

    @staticmethod
    def make_station_names(train_dates, test_dates, stations, station_layout,
                           feature):
        """ Takes the dates of the training dates, test dates, and the station
        names and returns the X train and X test data.
        """

        results = Engineer.parse_parameters(station=stations)
        stations = results['station']

        train_index = Engineer.create_index(
            train_dates=train_dates, station=stations)
        test_index = Engineer.create_index(
            test_dates=test_dates, station=stations)

        trainX = train_index
        testX = test_index

        trainX['station_name'] = train_index.loc[:, 'station']
        testX['station_name'] = test_index.loc[:, 'station']
        # start by resetting the index so that these can be more easily
        trainX = trainX.reset_index()
        testX = testX.reset_index()
        train_dum = pd.get_dummies(trainX['station_name'], prefix='stat')
        test_dum = pd.get_dummies(testX['station_name'], prefix='stat')
        trainX = pd.merge(trainX, train_dum.iloc[:, :-1], left_on='index',
                          right_index=True)
        testX = pd.merge(testX, test_dum.iloc[:, :-1], left_on='index',
                         right_index=True)
        trainX.drop('station_name', axis=1, inplace=True)
        testX.drop('station_name', axis=1, inplace=True)
        trainX.rename(columns={0: feature['type']}, inplace=True)
        testX.rename(columns={0: feature['type']}, inplace=True)
        trainX.drop('index', axis=1, inplace=True)
        testX.drop('index', axis=1, inplace=True)

        # list the columns that are part of the index
        # subtract off the data columns (remember we lose one in getting
        # dummies)
        train_cols = list(trainX.columns)[:(1 - len(stations))]
        test_cols = list(testX.columns)[:(1 - len(stations))]

        # we don't want the indices to be columns, so use all of the columns
        # for indices
        trainX = trainX.set_index(train_cols)
        testX = testX.set_index(test_cols)

        if (station_layout):
            train_diff_cols = {'train_dates', 'station'}
            test_diff_cols = {'test_dates', 'station'}
        else:
            train_diff_cols = {'train_dates'}
            test_diff_cols = {'test_dates'}

        # With the known columns move the other indices to the top
        trainX = trainX.unstack(list(
            set(train_cols).difference(train_diff_cols)))
        testX = testX.unstack(list(
            set(test_cols).difference(test_diff_cols)))

        trainX = trainX.swaplevel(0, 1, axis=0).sort_index()
        testX = testX.swaplevel(0, 1, axis=0).sort_index()
        return trainX, testX

    @staticmethod
    def fill_default_val(param):
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
            return range(0, 11)
        elif (param == 'times'):
            return range(12, 25, 3)
        elif (param == 'lats'):
            return range(31, 40)
        elif (param == 'longs'):
            return range(254, 270)
        elif (param == 'var'):
            return [
                'dswrf_sfc', 'dlwrf_sfc', 'uswrf_sfc', 'ulwrf_sfc',
                'ulwrf_tatm', 'pwat_eatm', 'tcdc_eatm', 'apcp_sfc', 'pres_msl',
                'spfh_2m', 'tcolc_eatm', 'tmax_2m', 'tmin_2m', 'tmp_2m',
                'tmp_sfc']
        elif (param == 'train_dates'):
            return ['1994-01-01', '2007-12-31']
        elif (param == 'test_dates'):
            return ['2008-01-01', '2012-11-30']
        else:
            return None

    @staticmethod
    def make_index_num(**kwargs):
        """ Take a list of values and set them to their index number if it is
        not already the default index numbers
        """

        # iterate over all passed items
        indexed_args = {}
        for key, val in kwargs.iteritems():
            value = np.array(val)
            if (key == 'times'):
                value = (value - 12)//3
            elif (key == 'lats'):
                value = value - 31
            elif (key == 'longs'):
                value = value - 254
            elif (key == 'train_dates'):
                value = [value[0], value[-1]]
            elif (key == 'test_dates'):
                value = [value[0], value[-1]]
            indexed_args[key] = value
        return indexed_args

    @staticmethod
    def fill_default(**kwargs):
        """ Fill all the values in the dictionary with value 'all' with the
        appropriate default values.
        """

        # Start building the return dictionary
        return_args = {}

        # iterate over all of the passed items
        for key, value in kwargs.iteritems():
            if ('all' in value):
                return_args[key] = Engineer.fill_default_val(key)
            else:
                return_args[key] = value
            if ((key == 'train_dates') or (key == 'test_dates')):
                return_args[key] = np.arange(
                    np.datetime64(value[0]), np.datetime64(value[1]) + 1,
                    dtype='datetime64')
        return return_args

    @staticmethod
    def parse_parameters(**kwargs):
        """ Take a set of parameters and fill it with defaults. Then,
        convert the values to indices and return the passed parameters.
        """

        return_args = Engineer.fill_default(**kwargs)
        indexed_args = Engineer.make_index_num(**return_args)

        return indexed_args

    @staticmethod
    def create_index(**kwargs):
        """ Create an index based on the input X parameters and the
        engineered features.
        """
        param_dict = Engineer.fill_default(**kwargs)
        col_names = param_dict.keys()
        new_indices = pd.DataFrame(list(
            itertools.product(*param_dict.values())))
        new_indices.columns = col_names
        return new_indices
