#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" solar-admin
 An administrative script the solar energy prediction kaggle competition

 Author:   Joel Piper <joelmpiper [at] gmail.com>
 Created: Saturday, January 9, 2016

An administrative script for the Kaggle solar energy competition
"""

# Imports

import sys
import numpy as np
from solar.wrangle.wrangle import SolarData
from solar.analyze.model import Model
from sklearn.linear_model import Ridge
from solar.report.submission import Submission


def main(*argv):

    Xtrain_dir = 'solar/data/kaggle_solar/train/'
    Xtest_dir = 'solar/data/kaggle_solar/test'
    ytrain_file = 'solar/data/kaggle_solar/train.csv'
    station_file = 'solar/data/kaggle_solar/station_info.csv'

    station = ['ACME', 'BEAV']
    train_dates = ['1994-01-01', '1994-03-01']
    test_dates = ['2008-01-01', '2008-01-05']
    station_layout = True
    var = ['uswrf_sfc', 'dswrf_sfc', ]

    model = [0]

    times = [21]

    default_grid = {'type': 'relative',
                    'axes': {'var': var, 'models': model, 'times': times,
                             'station': station}}
    just_grid = [default_grid]
    input_data = SolarData.load(Xtrain_dir, ytrain_file, Xtest_dir,
                                station_file, train_dates, test_dates, station,
                                station_layout, just_grid, 'local')

    error_formula = 'mean_absolute_error'
    cv_splits = 10

    model = Model.model(input_data, Ridge,
                        {'alpha': np.logspace(-3, 1, 8, base=10)}, cv_splits,
                        error_formula, 1, 'local', normalize=True)

    preds = Submission.make_submission_file(model, input_data[1],
                                            input_data[2], True, 'local')

    print(model.best_score_)
    print(preds)

if __name__ == '__main__':
    main(*sys.argv)
