""" This file will return inputs in dataframes that can be used for simple
    statistics and plotting.
    joelmpiper [at] gmail.com
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from solar.wrangle.wrangle import SolarData

class SolarDF(object):
    """ This returns the dataframes of the training X, y and testing
    X, and locations.
    """

    def __init__(self, trainX_file, trainy_file, testX_file, testy_file):
        self.trainX_file = trainX_file
        self.trainy_file = trainy_file
        self.testX_file = testX_file
        self.testy_file = testy
        self.solar_df = create_df( self.trainX_file, self.trainy_file,
                                  self.testX_file, self.testy_file)

    @staticmethod
    def create_df(trainX_dir, trainy_file, testX_dir, loc_file, **kwargs):
        """ Returns the X, y, and location data cut by the parameters specified.

            Take any list of dates, times, models, latitudes, longitudes,
            and predictive locations, as well as elevation ranges.
        """
        solar_dfs = SolarData.load_data_explore(trainX_dir, trainy_file,
                                                testX_dir, loc_file, **kwargs)
        return solar_dfs


if __name__ == '__main__':
    s = SolarDF(trainX_file='solar/data/kaggle_solar/train/',
               trainy_file='solar/data/kaggle_solar/train.csv',
               testX_file='solar/data/kaggle_solar/test/',
               loc_file='solar/data/kaggle_solar/station_info.csv')
    print s.trainX.shape
