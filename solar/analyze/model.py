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
sys.path.append(os.getcwd())
from solar.wrangle.wrangle import InputArray
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split
from sklearn import metrics

class Model(object):
    """ Load the solar data from the gefs and mesonet files """

    def __init__(self, benchmark=True, pickle=False,
                 input_pickle='solar/data/kaggle_solar/all_input.p',
                 input_data=None, model=Ridge):

        self.benchmark = benchmark
        self.pickle = pickle
        self.model = model
        self.input_pickle = input_pickle
        if(benchmark):
            self.data = self.model_benchmark(input_data)
        else:
            if(pickle):
                self.data = self.load_pickle()
            else:
                self.data = self.model_benchmark(input_data)

    def model_benchmark(self, input_data):


        def cv_loop(X, y, model, N):
            MAEs = 0
            for i in range(N):
                X_train, X_cv, y_train, y_cv = train_test_split(
                    X, y, test_size=.20, random_state = i*7)
                model.fit(X_train, y_train)
                preds = model.predict(X_cv)
                mae = metrics.mean_absolute_error(y_cv,preds)
                print "MAE (fold %d/%d): %f" % (i + 1, N, mae)
                MAEs += mae
            return MAEs/N


        if (self.input_pickle):
            trainX, trainY, times, testX = pickle.load(open(
                self.input_pickle,'rb'))
        elif (self.input_data):
            trainX, trainY, times, testX = self.input_data
        else:
            trainX, trainY, times, testX = InputArray().data
        # Gotta pick a scikit-learn model
        model = self.model(normalize=True)
        # Normalizing is usually a good idea

        print 'Finding best regularization value for alpha...'
        alphas = np.logspace(-3,1,8,base=10)
        # List of alphas to check
        #alphas = np.array(( 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ))
        alphas = np.array((0.3, 0.2))
        maes = []
        for alpha in alphas:
            model.alpha = alpha
            mae = cv_loop(trainX,trainY,model,10)
            maes.append(mae)
            print 'alpha %.4f mae %.4f' % (alpha,mae)
        best_alpha = alphas[np.argmin(maes)]
        print ('Best alpha of %s with mean average error of %s'
                % (best_alpha,np.min(maes)))

        print 'Fitting model with best alpha...'
        model.alpha = best_alpha
        model.fit(trainX,trainY)

        self.model = model
        pickle.dump((trainX, trainY, times),
                    open("solar/data/kaggle_solar/model.p","wb"))

        return


    def load_pickle(self, pickle_dir='solar/data/kaggle_solar/'):
        return pickle.load(open(pickle_dir + '/model.p','rb'))


if __name__ == '__main__':
    model = Model()
    #trainX, trainY, times = (data.load('solar/data/kaggle_solar/train/',
    #                        'solar/data/kaggle_solar/', 'all'))
    print Model
