#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Test the functionality of building the y data from file
"""

from solar.wrangle.wrangle import SolarData

trainy_file = "solar/data/kaggle_solar/train.csv"
loc_file = "solar/data/kaggle_solar/station_info.csv"
dates = ['1994-01-01', '1994-02-03']
stations = ['ACME', 'BEAV']
sta_layout = True

trainy, locy = SolarData.load_trainy(trainy_file, loc_file, dates,
                                     stations, sta_layout)
assert (trainy.loc['1994-02-03', 'ACME'][0] == 14489400)
assert (trainy.loc['1994-01-01', 'ACME'][0] == 12384900)
assert (trainy.loc['1994-01-01', 'BEAV'][0] == 10116900)
assert (trainy.loc['1994-02-03', 'BEAV'][0] == 12247800)
assert (trainy.columns == 'total_solar')
