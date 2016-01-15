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
from solar.wrangle.wrangle import InputArray
from solar.analyze.model import Model
from solar.report.submission import Submission


def main(*argv):
    print("Test!")
    print(os.listdir('./'))
    solar_array = InputArray()
    model = Model(input_data=solar_array)
    submission = Submission(input_model=model, input_data=solar_array)

    print submission
    #print model

if __name__ == '__main__':
    main(*sys.argv)
