#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This includes the beat-the-benchmark files that are associated with
the initial data wrangling. See the URL to the forum post with the code:
https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest/forums/t/5282/beating-the-benchmark-in-python-2230k-mae
by Alec Radford.
"""

import numpy as np
import cPickle as pickle
import datetime
import logging
from sklearn import metrics
from solar.wrangle.wrangle import SolarData
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.grid_search import GridSearchCV


class Model(object):
    """ Load the solar data from the gefs and mesonet files """

    def __init__(self, benchmark=False, pickle=False,
                 input_pickle='solar/data/kaggle_solar/all_input.p',
                 input_data=None, model=Ridge,
                 param_dict={'alpha': np.logspace(-3, 1, 8, base=10)},
                 cv_splits=10, err_formula=mean_absolute_error,
                 **model_params):

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
                self.data = self.model(input_data, model, param_dict,
                                       cv_splits, err_formula,
                                       **model_params)

    @staticmethod
    def model_from_pickle(pickle_name, model_name, param_dict, cv_splits,
                          err_formula, n_jobs, write, **model_params):
        """ Run the model function from a pickled input
        """
        if (write == 'extern'):
            pickle_file = '/Volumes/Seagate Backup Plus Drive/' + pickle_name
        elif (write == 'local'):
            pickle_file = 'solar/data/kaggle_solar/inputs/' + pickle_name
        elif (write == 's3'):
            pickle_file = '/home/ec2-user/mount_point/data/kaggle_solar/' + \
                'inputs/' + pickle_name
        else:
            print("None")

        input_data = pickle.load(open(pickle_file, 'rb'))
        return Model.model(input_data, model_name, param_dict, cv_splits,
                           err_formula, n_jobs, write, **model_params)

    @staticmethod
    def model(input_data, model_name, param_dict, cv_splits,
              err_formula, n_jobs, write, **model_params):
        """ Takes input data which contains X and y training data, the
        name of the sklearn model, a dictionary of parameters, the number of
        cross_validation splits, the forumula to calulate errors in the fit,
        and any additional model parameters and creates a fitted model that
        can be used for predictions.
        """

        file_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if (write == 's3'):
            fh = logging.FileHandler('/home/ec2-user/mount_point/log/log_' +
                                     file_time + '.log')
        else:
            fh = logging.FileHandler('log/log_' + file_time + '.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.info('Started building model')
        logger.info('Train input columns: %s' % (input_data[0].columns))
        logger.info('Train input indices: %s' % (input_data[0].index.levels))
        logger.info('Test input indices: %s' % (input_data[2].index.levels))
        logger.info('Model name: %s' % (model_name))
        logger.info('Params: %s' % (param_dict))
        logger.info('CV splits: %s' % (cv_splits))
        logger.info('Error formula: %s' % (err_formula))
        logger.info('Model params: %s' % (model_params))
        # The testX data is unneeded to create the model
        trainX, trainy, __, __ = input_data

        model = model_name(**model_params)

        # minimize mean absolute error (will be the check for the results)
        grid = GridSearchCV(model, param_dict, cv=cv_splits,
                            scoring=err_formula, n_jobs=n_jobs)

        grid.fit(trainX, trainy)

        logger.info('Best score: %s' % (grid.best_score_))
        logger.info('Best params: %s' % (grid.best_params_))
        logger.info('Best estimator: %s' % (grid.best_estimator_))

        if ((write == 'local') or (write == 'extern')):
            pickle.dump((grid),
                        open("solar/data/kaggle_solar/models/model_" +
                             file_time + ".p", "wb"))
        elif (write == 's3'):
            pickle.dump((grid),
                        open("/home/ec2-user/mount_point/data/kaggle_solar/" +
                             "models/model_" + file_time + ".p", "wb"))
        else:
            print "None"

        return grid

    @staticmethod
    def cv_loop(X, y, model, N):
        """ Takes an X and y and performs a cross-validated fitting of the
        training data.

        This function is nearly exactly the same as the version in
        beat the benchmark.
        """

        MAEs = 0
        for i in range(N):
            X_train, X_cv, y_train, y_cv = train_test_split(
                X, y, test_size=.20, random_state=(i * 7))
            model.fit(X_train, y_train)
            preds = model.predict(X_cv)
            mae = metrics.mean_absolute_error(y_cv, preds)
            print "MAE (fold %d/%d): %f" % (i + 1, N, mae)
            MAEs += mae
        return MAEs/N

    def model_benchmark(self, input_data):
        """ Takes input data and the Model class object and stores the
        fitted model.

        This is modified from beat the benchmark. I added some
        pickle functionality because I was testing it and didn't want
        to always run through the entire analysis. Otherwise, it is
        minimally changed.
        """

        if (self.input_pickle):
            trainX, trainY, testX = pickle.load(open(
                self.input_pickle, 'rb'))
        elif (self.input_data):
            trainX, trainY, testX = self.input_data
        else:
            trainX, trainY, testX = SolarData().data
        # Gotta pick a scikit-learn model
        model = self.model(normalize=True)
        # Normalizing is usually a good idea

        print 'Finding best regularization value for alpha...'
        alphas = np.logspace(-3, 1, 8, base=10)
        # List of alphas to check
        alphas = np.array((0.3, 0.2))
        maes = []
        for alpha in alphas:
            model.alpha = alpha
            mae = Model.cv_loop(trainX, trainY, model, 10)
            maes.append(mae)
            print 'alpha %.4f mae %.4f' % (alpha, mae)
        best_alpha = alphas[np.argmin(maes)]
        print ('Best alpha of %s with mean average error of %s'
               % (best_alpha, np.min(maes)))

        print 'Fitting model with best alpha...'
        model.alpha = best_alpha
        model.fit(trainX, trainY)

        self.model = model
        pickle.dump((model),
                    open("solar/data/kaggle_solar/model.p", "wb"))

        return

    def load_pickle(self, pickle_dir='solar/data/kaggle_solar/'):
        return pickle.load(open(pickle_dir + '/model.p', 'rb'))


if __name__ == '__main__':
    model = Model()
    print Model
