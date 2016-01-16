#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This includes the beat-the-benchmark files that are associated with
the initial data wrangling. See the URL to the forum post with the code:
https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest/forums/t/5282/beating-the-benchmark-in-python-2230k-mae
by Alec Radford.
"""

import numpy as np
import cPickle as pickle
import sys
import os
import csv
sys.path.append(os.getcwd())
from solar.wrangle.wrangle import SolarData
from solar.analyze.model import Model
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split
from sklearn import metrics

class Submission(object):
    """ Load the solar data from the gefs and mesonet files """

    def __init__(self, benchmark=True, pickle=False,
                 input_model=None, input_data=None,
                 model_pickle='',
                 input_pickle=''):

        self.benchmark = benchmark
        self.pickle = pickle
        self.input_data = input_data
        self.input_pickle = input_pickle
        if (pickle):
            self.load_pickle()
        else:
            self.get_model(input_model)

    def get_model(self, input_model):


        def save_submission(preds,data_dir):
            fexample = open(os.path.join(data_dir,'sampleSubmission.csv'))
            fout = open('sample.csv','wb')
            fReader = csv.reader(fexample,delimiter=',', skipinitialspace=True)
            fwriter = csv.writer(fout)
            for i,row in enumerate(fReader):
                if i == 0:
                    fwriter.writerow(row)
                else:
                    row[1:] = preds[i-1]
                    fwriter.writerow(row)
            fexample.close()
            fout.close()


        if (self.input_pickle):
            model = pickle.load(open(input_model,'rb'))
            __, __, __, testX = pickle.load(open(input_data,'rb'))
        elif (input_model):
            model = input_model.model
            __, __, __, testX = self.input_data.data
        else:
            model = Model().model
            __, __, __, testX = SolarData().data

        print 'Predicting...'

        preds = model.predict(testX)

        print 'Saving to csv...'
        save_submission(preds, 'solar/data/kaggle_solar/')

        self.preds =preds
        pickle.dump(preds,
                    open("solar/data/kaggle_solar/preds.p","wb"))

        return


    def load_pickle(self, pickle_dir='solar/data/kaggle_solar/'):
        return pickle.load(open(pickle_dir + '/preds.p','rb'))


if __name__ == '__main__':
    preds = Submission(input_pickle=False)
    #trainX, trainY, times = (data.load('solar/data/kaggle_solar/train/',
    #                        'solar/data/kaggle_solar/', 'all'))
    print preds
