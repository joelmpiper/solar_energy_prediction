""" This file returns specified subsets of all relevant data as an input array.
    This can reduce the file sizes associated with all input, training, and
    test sets.
    joelmpiper [at] gmail.com
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime as pydt

TEST = 0
class Subset(object):
    """ This returns the appropriate complementary training X, y and testing
    X and y.
    """

    def __init__(self, trainX_file, trainy_file, testX_file, testy_file):
        self.trainX_file = trainX_file
        self.trainy_file = trainy_file
        self.testX_file = testX_file
        self.testy_file = testy
        self.trainX, self.trainy, self.testX, self.testy, self.loc = subset(
            self.trainX_file, self.trainy_file, self.testX_file,
            self.testy_file)


    @staticmethod
    def subset(trainX_dir, trainy_file, testX_dir, loc_file, **kwargs):
        """ Returns the X, y, and location data cut by the parameters specified.

            Take any list of dates, times, models, latitudes, longitudes,
            and predictive locations, as well as elevation ranges.
        """

        X_params = {}
        y_params = {}
        for key, value in kwargs.iteritems():
            if key in ['train_dates', 'station', 'lats', 'longs', 'elevs']:
                y_params[key] = value
            if key in ['train_dates', 'test_dates', 'var', 'models', 'lats',
                       'longs', 'elevs']:
                X_params[key] = value
        trainX, testX = Subset.create_Xdata(trainX_dir, testX_dir, **X_params)
        trainy, loc = Subset.create_ydata(trainy_file, loc_file, **y_params)
        return trainX, trainy, testX, loc


    @staticmethod
    def create_ydata(train_file, loc_file, train_dates=[], station=[], lats=[],
                     longs=[], elevs=[]):
        """ Returns a dataframes of the y data and the location data.

            Create dataframes by reading in the training and location files;
            tidy the training set by moving the location code from a column to
            to a row.
        """
        train_data = pd.read_csv(train_file, index_col='Date',
                                 parse_dates=['Date'])
        rot_df = pd.DataFrame(train_data.unstack().reset_index()[:])
        rot_df.rename(columns={'level_0':'location','Date':'date',
                            0:'total_solar'}, inplace=True)
        rot_df.set_index('date',inplace=True)
        loc_df = pd.read_csv(loc_file, index_col='stid')

        loc_df['elon'] = loc_df['elon'] + 360
        loc_df.rename(columns={'nlat':'lat','elon':'lon'}, inplace=True)
        loc_df.index.names = ['location']

        # if we are making any specifications on the output data, then
        # reduce what is returned to conform with those specifications
        if(station or lats or longs or elevs or train_dates):
            rot_df, loc_df = Subset.reduced_training(
                rot_df, loc_df, train_dates, station, [min(lats), max(lats)],
                [min(longs), max(longs)], elevs)
        return rot_df, loc_df


    @staticmethod
    def reduced_training(rot_df, loc_df, date, station, lat, lon, elev):
        """ Returns the y output and location data subsets given the parameters.

            Take the output data and location data and return the reduced set.
        """

        # check if there are a reduced list of stations
        if (station):
            mod_train = rot_df[rot_df['location'].isin(station)]
            mod_loc = loc_df.loc[station,:]
        # if there are a list of stations, then we don't want to choose the
        # stations by any other parameters (so else not separate ifs)
        else:
            mod_train = rot_df
            mod_loc = loc_df
            if (lat):
                if (TEST > 200):
                    print "Mod Train Lat"
                    print(mod_train)
                    print "Mod Loc Lat"
                    print(mod_loc)
                mod_train, mod_loc = Subset.cut_var(mod_train, mod_loc, lat,
                                                    'lat', rot_df.columns,
                                                    loc_df.columns)
            if (lon):
                if (TEST > 0):
                    print "Mod Train Lon"
                    print(mod_train)
                    print "Mod Loc Lon"
                    print(mod_loc)
                mod_train, mod_loc = Subset.cut_var(mod_train, mod_loc, lon,
                                                    'lon', rot_df.columns,
                                                    loc_df.columns)
            if (elev):
                mod_train, mod_loc = Subset.cut_var(mod_train, mod_loc, elev,
                                                    'elev', rot_df.columns,
                                                    loc_df.columns)
        # on the other hand, the date cuts can be paired with any of the other
        # cuts and only affects output data (not location)
        if (date):
            mod_train.sort_index(inplace=True)
            mod_train = mod_train.loc[date[0]:date[1],:].reset_index()
            mod_train = mod_train.sort_values(['location','date'])
            mod_train = mod_train.set_index('date')
        return mod_train, mod_loc


    @staticmethod
    def cut_var(train, loc, var, var_name, train_split, loc_split):
        """ Take output dataframes and reduce the single variable. """

        if (TEST > 0):
            print("Train in cut_var:")
            print(train)
        combined = train.merge(loc, left_on='location',
                                    right_index=True)

        if (TEST > 0):
            print("Combined after first merge:")
            print(combined)
        combined = combined[(combined[var_name] >= var[0]) &
                            (combined[var_name] <= var[1])]
        if (TEST > 0):
            print("Combined after first cut:")
            print(combined)
        mod_train = combined[train_split].drop_duplicates()
        if (TEST > 0):
            print("Mod Train after resplitting:")
            print(mod_train)
        mod_loc = combined[loc_split.union(['location'])]
        if (TEST > 0):
            print("Modified locatin:")
            print(mod_loc)
            print("Train:")
            print(mod_train)
            print("Combined:")
            print(combined)
        mod_loc = mod_loc.reset_index().drop('date', axis=1)
        mod_loc = mod_loc.drop_duplicates().set_index('location')

        return mod_train, mod_loc


    @staticmethod
    def create_Xdata(trainX_dir, testX_dir, var=[], train_dates=[],
                     test_dates=[], models=[], lats=[], longs=[],
                     times=[], elevs=[]):
        """ Return an np array of X data with the given cuts.
        """
        # The default is all of the files, so load them all if none are
        # specified
        if ((not var) or var[0] == 'all'):
            var = [
                'dswrf_sfc','dlwrf_sfc','uswrf_sfc','ulwrf_sfc',
                'ulwrf_tatm','pwat_eatm','tcdc_eatm','apcp_sfc','pres_msl',
                'spfh_2m','tcolc_eatm','tmax_2m','tmin_2m','tmp_2m','tmp_sfc'
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
        for i,f in enumerate(var):
            if i == 0:
                X = Subset.cutX(input_dir + '/' + var[i] + substr,
                                dates, models, lats, longs, times, elevs)
            else:
                X_new = Subset.cutX(input_dir + '/' + var[i] + substr,
                                    dates, models, lats, longs, times,
                                    elevs)
                X = np.hstack((X,X_new))
            if (TEST > 0):
                print("Iteration " + str(i) + " complete.")
                print("Variable: " + f)
        return X


    @staticmethod
    def cutX(file_name, input_date, models=None,
             lats=None, longs=None,
             times=None, elevs=None):
        """ Load a single nc file and return a subset of that file in an
        np array.
        """

        X = nc.Dataset(file_name,'r+').variables.values()

        # set defaults
        begin_date = 0
        end_date = len(X[1][:])
        bottom_elev = -100
        top_elev = 30000
        # if date values are provided revice the ranges
        if (input_date):
            dates = X[1][:]
            begin_date, end_date = Subset.check_dates(dates, input_date)

        if (not models):
            models = np.arange(0,11)
        if (not lats):
            lats = np.arange(31,40)
        if (not longs):
            longs = np.arange(254,270)
        if (not times):
            times = np.arange(15,25,3)
        if (elevs):
            bottom_elev = elevs[0]
            top_elev = elevs[1]
        if (TEST > 0):
            print("Begin date: " + str(begin_date))
            print("End date: " + str(end_date))
            print("Models: " + ", ".join([str(mod) for mod in models]))
        # Five times from 15 to 24 by 3
        time_mod = (np.array(times) - 15)//3
        if (TEST > 0):
            print(time_mod)
        # First latitude is at 31 and by one degree from there
        lat_mod = np.array(lats) - 31
        if (TEST > 0):
            print (lat_mod)
        # First longitude at 254 and by one degree from there
        long_mod = np.array(longs) - 254
        if (TEST > 0):
            print (long_mod)
        X = X[-1][begin_date:end_date,models,time_mod,lat_mod,long_mod]
        if (TEST > 0):
            print("Subset complete")

        # drop into two dimensions so variables can be stacked
        # it can later be manipulated for learning or exploring
	    X = X.reshape(X.shape[0],np.prod(X.shape[1:])) 					 # Reshape into (n_examples,n_features)

        return X


    @staticmethod
    def check_dates(dates, input_date):
        """ Take a list of dates, a beginning date, and an ending date
        and return the indices needed to extract information from the
        np array.
        """

        date_range = [pydt.strptime(str(myDate), "%Y%m%d%H")
                    for myDate in dates]
        begin_date = date_range.index(
            pydt.strptime(input_date[0],"%Y-%m-%d"))
        if(TEST > 0):
            print begin_date
        # Add a 1 at the end to include the date in the range
        end_date = date_range.index(pydt.strptime(
            input_date[1],"%Y-%m-%d")) + 1
        if(TEST > 0):
            print end_date

        return begin_date, end_date

if __name__ == '__main__':
    s = Subset(trainX_file='solar/data/kaggle_solar/train/',
               trainy_file='solar/data/kaggle_solar/train.csv',
               testX_file='solar/data/kaggle_solar/test/',
               loc_file='solar/data/kaggle_solar/station_info.csv')
    print s.trainX.shape
