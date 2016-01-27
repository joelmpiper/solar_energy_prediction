""" This file returns engineered features for all of the individual points on
specific days.

    joelmpiper [at] gmail.com
"""

import numpy as np
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
    def engineer(trainX_dir, yfile, testX_dir, locfile,
                 Xparams, yparams, features):

        y_df, loc_df = Subset.create_ydata(yfile, locfile, **yparams)

        if ('grid' in features):
            trainX, testX = Engineer.create_grid(trainX_dir, testX_dir,
                                                 Xparams, loc_df)
        return trainX, y_df, testX, loc_df
