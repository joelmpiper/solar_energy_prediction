#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This includes the beat-the-benchmark files that are associated with
the initial data wrangling. See the URL to the forum post with the code:
https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest/forums/t/5282/beating-the-benchmark-in-python-2230k-mae
by Alec Radford.
"""

import logging
import pickle
import os
import csv
import datetime
from solar.wrangle.wrangle import SolarData
from solar.analyze.model import Model


class Submission(object):
    """ Load the solar data from the gefs and mesonet files """

    def __init__(self, benchmark=False, pickle=False,
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
            if (benchmark):
                self.get_benchmark_model(input_model)
            else:
                Submission.make_submission_file(input_model, input_data[1],
                                                input_data[2])

    @staticmethod
    def submit_from_pickle(pickle_model, pickle_data, station_format, write):

        """ Run the model function from a pickled input
        """
        if ((write == 'extern') or (write == 'local')):
            file_base = 'solar/'
        elif (write == 's3'):
            file_base = '/home/ec2-user/mount_point/'
        model_file = file_base + '/data/kaggle_solar/models/' + pickle_model
        model = pickle.load(open(model_file, 'rb'))
        data_file = file_base + '/data/kaggle_solar/inputs/' + pickle_data

        __, trainy, testX, __ = pickle.load(open(data_file, 'rb'))

        return Submission.make_submission_file(model, trainy, testX,
                                               station_format, write)

    @staticmethod
    def make_submission_file(model, trainy, testX, station_format, write):
        """ Take a model and list of x and return a csv with predictions.

        Make submission file containing y predictions based on the the given
        input model and test X.
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

        logger.info('Started building submission file')
        preds = model.predict(testX)
        if (station_format):
            print(testX)
            print(trainy)
            num_rows = len(testX.index.levels[0])
            trainy = trainy.unstack('location')
            preds = preds.reshape(num_rows, trainy.shape[1])
            column_names = trainy['total_solar'].columns
        else:
            column_names = trainy.columns

        if ((write == 'extern') or (write == 'local')):
            file_base = 'solar/'
        elif (write == 's3'):
            file_base = '/home/ec2-user/mount_point/'
        with open(file_base + '/submissions/submission_' +
                  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") +
                  '.csv', 'wb') as fout:

            fwriter = csv.writer(fout)
            fwriter.writerow(['Date'] + list(column_names))
            for i, val in enumerate(testX.index.levels[0]):
                row = [val.strftime("%Y%m%d")]
                row = row + [x[0] for x in zip(preds[i])]
                fwriter.writerow(row)

        logger.info('Finished building submission file')

        pickle.dump((preds),
                    open(file_base + 'data/kaggle_solar/submissions/submit_' +
                         file_time + '.p', 'wb'))
        return preds

    def get_benchmark_model(self, input_model, input_data):

        def save_submission(preds, data_dir):
            fexample = open(os.path.join(data_dir, 'sampleSubmission.csv'))
            fout = open('sample.csv', 'wb')
            fReader = csv.reader(fexample, delimiter=',', skipinitialspace=True)
            fwriter = csv.writer(fout)
            for i, row in enumerate(fReader):
                if i == 0:
                    fwriter.writerow(row)
                else:
                    row[1:] = preds[i-1]
                    fwriter.writerow(row)
            fexample.close()
            fout.close()

        if (self.input_pickle):
            model = pickle.load(open(input_model, 'rb'))
            __, __, __, testX = pickle.load(open(input_data, 'rb'))
        elif (input_model):
            model = input_model.model
            __, __, __, testX = self.input_data.data
        else:
            model = Model().model
            __, __, __, testX = SolarData().data

        print('Predicting...')

        preds = model.predict(testX)

        print('Saving to csv...')
        save_submission(preds, 'solar/data/kaggle_solar/')

        self.preds = preds
        pickle.dump(preds,
                    open("solar/data/kaggle_solar/preds.p", "wb"))

        return

    def load_pickle(self, pickle_dir='solar/data/kaggle_solar/'):
        return pickle.load(open(pickle_dir + '/preds.p', 'rb'))


if __name__ == '__main__':
    preds = Submission(input_pickle=False)
#    trainX, trainY, times = (data.load('solar/data/kaggle_solar/train/',
#                            'solar/data/kaggle_solar/', 'all'))
    print(preds)
