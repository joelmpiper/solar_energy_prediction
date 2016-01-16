#!/usr/bin/env python
 # -*- coding: utf-8 -*-

# solar-admin
# An administrative script the solar energy prediction kaggle competition
#
# Author:   Joel Piper <joelmpiper [at] gmail.com>
# Created: Saturday, January 9, 2016
#

"""
An administrative script for the Kaggle solar energy competition
"""

##########################################################################
## Imports
##########################################################################

import os
import sys
import csv
import argparse
sys.path.append(os.getcwd())
from solar.wrangle.wrangle import SolarData
from solar.analyze.model import Model
from solar.report.submission import Submission


def main(*argv):
    print("Test!")
    print(os.listdir('./'))
    train_dates = ['2005-07-29','2005-08-02']
    test_dates = ['2008-12-29', '2009-01-05']
    #var = ['dswrf_sfc', 'uswrf_sfc', 'spfh_2m', 'ulwrf_tatm']
    var = ['dswrf_sfc']
    models = [0]
    times = [12]
    lats = range(35,37)
    longs = range(260,262)
    solar_array = SolarData(var=var, models=models, lats=lats,
                            longs=longs, train_dates=train_dates,
                            test_dates=test_dates)
    model = Model(input_data=solar_array)
    submission = Submission(input_model=model, input_data=solar_array)

    print submission
    #print model

if __name__ == '__main__':
    main(*sys.argv)
