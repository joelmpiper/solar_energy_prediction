""" This file returns engineered features for all of the individual points on
specific days.


    joelmpiper [at] gmail.com
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from solar.wrangle.subset import Subset


TEST = 0
class Engineer(object):
    """ This returns a y created from the subset file and an X file created
    by engineering features for each individual data point.
    """

    def __init__(self, trainX_file):
        self.trainX_file = trainX_file

    @staticmethod
    def create_grid(trainX_dir, testX_dir, Xparams, loc_df):
        """ take the station north, east, south, and west of each of
        the given points, as well as any other specified cuts and
        return the new file.
        """

        for i, sta in enumerate(loc_df.index):
            stlat = loc_df.loc[sta].lat
            stlong = loc_df.loc[sta].lon
            lats = range(int(np.floor(stlat)), int(np.ceil(stlat)) + 1)
            longs = range(int(np.floor(stlong)), int(np.ceil(stlong)) + 1)
            Xparams['lats'] = lats
            Xparams['longs'] = longs
            sta_train, sta_test = Subset.create_Xdata(
                trainX_dir, testX_dir, **Xparams)
            if (i == 0):
                train_grid = sta_train
                test_grid = sta_test
            else:
                train_grid = np.vstack((train_grid, sta_train))
                test_grid = np.vstack((test_grid, sta_test))

        return train_grid, test_grid


    @staticmethod
    def engineer(trainX_dir, yfile, testX_dir, locfile,
                 Xparams, yparams, features):

        y_df, loc_df = Subset.create_ydata(yfile, locfile, **yparams)

        if ('grid' in features):
            trainX, testX = Engineer.create_grid(trainX_dir, testX_dir,
                                        Xparams, loc_df)
        """
        if ('mean_val' in features):
            create_mean(trainX_dir, y_file, params)
        """
        return trainX, y_df, testX, loc_df
