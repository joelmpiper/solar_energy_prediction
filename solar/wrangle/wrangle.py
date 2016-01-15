#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This includes the beat-the-benchmark files that are associated with
the initial data wrangling. See the URL to the forum post with the code:
https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest/forums/t/5282/beating-the-benchmark-in-python-2230k-mae
by Alec Radford.



"""

import os
import sys
import netCDF4 as nc
import numpy as np
import cPickle as pickle
sys.path.append(os.getcwd())
from solar.wrangle.subset import Subset

class SolarData(object):
    """ Load the solar data from the gefs and mesonet files """

    def __init__(self, benchmark=False, pickle=False,
                 trainX_dir='solar/data/kaggle_solar/train/',
                 testX_dir='solar/data/kaggle_solar/test/',
                 meso_dir='solar/data/kaggle_solar/',
                 trainy_file='solar/data/kaggle_solar/train.csv',
                 loc_file='solar/data/kaggle_solar/station_info.csv',
                 files_to_use=['dswrf_sfc'], **kwargs):
        self.benchmark = benchmark
        self.pickled = pickle
        if(benchmark):
            self.data = self.load_benchmark(trainX_dir, testX_dir, meso_dir,
                                            files_to_use)
        else:
            if(pickle):
                self.data = self.load_pickle()
            else:
                self.data = self.load(
                    trainX_dir, trainy_file, testX_dir, loc_file, **kwargs)

    def load_benchmark(self, gefs_train, gefs_test, meso_dir, files_to_use):
        """ Returns numpy arrays, of the training data """

        '''
        Loads a list of GEFS files merging them into model format.
        '''
        def load_GEFS_data(directory,files_to_use,file_sub_str):
            for i,f in enumerate(files_to_use):
                if i == 0:
                    X = load_GEFS_file(directory,files_to_use[i],file_sub_str)
                else:
                    X_new = load_GEFS_file(directory,files_to_use[i],file_sub_str)
                    X = np.hstack((X,X_new))
            return X

        '''
        Loads GEFS file using specified merge technique.
        '''
        def load_GEFS_file(directory,data_type,file_sub_str):
            print 'loading',data_type
            path = os.path.join(directory,data_type+file_sub_str)
            #X = nc.Dataset(path,'r+').variables.values()[-1][:,:,:,3:7,3:13] # get rid of some GEFS points
            X = nc.Dataset(path,'r+').variables.values()[-1][:,:,:,:,:]
            #X = X.reshape(X.shape[0],55,4,10) 								 # Reshape to merge sub_models and time_forcasts
            X = np.mean(X,axis=1) 											 # Average models, but not hours
            X = X.reshape(X.shape[0],np.prod(X.shape[1:])) 					 # Reshape into (n_examples,n_features)
            return X

        '''
        Load csv test/train data splitting out times.
        '''
        def load_csv_data(path):
            data = np.loadtxt(path,delimiter=',',dtype=float,skiprows=1)
            times = data[:,0].astype(int)
            Y = data[:,1:]
            return times,Y

        if files_to_use == 'all':
            files_to_use = ['dswrf_sfc','dlwrf_sfc','uswrf_sfc','ulwrf_sfc',
                    'ulwrf_tatm','pwat_eatm','tcdc_eatm','apcp_sfc','pres_msl',
                    'spfh_2m','tcolc_eatm','tmax_2m','tmin_2m','tmp_2m','tmp_sfc']
        train_sub_str = '_latlon_subset_19940101_20071231.nc'
        test_sub_str = '_latlon_subset_20080101_20121130.nc'

        print 'Loading training data...'
        trainX = load_GEFS_data(gefs_train,files_to_use,train_sub_str)
        times,trainY = load_csv_data(os.path.join(meso_dir,'train.csv'))
        print 'Training data shape',trainX.shape,trainY.shape

        testX = load_GEFS_data(gefs_test,files_to_use,test_sub_str)
        pickle.dump((trainX, trainY, times, testX),
                    open("solar/data/kaggle_solar/all_input.p","wb"))
        return trainX, trainY, times, testX


    @staticmethod
    def load_data_explore(trainX_dir, trainy_file, testX_dir, loc_file,
                          **kwargs):
        __, train_y, __, loc= Subset.subset(trainX_dir, trainy_file, testX_dir,
                                            loc_file, **kwargs)

        return train_y, loc


    @staticmethod
    def load_data_learn(trainX_dir, trainy_file, testX_dir, loc_file, **kwargs):
        train_X, train_y, test_X, __ = Subset.subset(
            trainX_dir, trainy_file, testX_dir, loc_file, **kwargs)

        train_y.reset_index(inplace=True)
        train_y = train_y.pivot(index='date', columns='location',
                                values='total_solar')
        return train_X, train_y, test_X


    @staticmethod
    def load(trainX_dir, trainy_file, testX_dir, loc_file, **kwargs):
        """ Returns numpy arrays, of the training data """

        trainX = load_GEFS_data(gefs_train,files_to_use,train_sub_str)
        trainX, testX, trainy = SolarData.load_data_learn(
            trainX_dir, trainy_file, testX_dir, loc_file, **kwargs)
        print 'Training data shape',trainX.shape,trainy.shape

        pickle.dump((trainX, trainy, testX),
                    open("solar/data/kaggle_solar/all_input.p","wb"))
        return trainX, trainy, testX

    def load_pickle(self, pickle_dir='solar/data/kaggle_solar/'):
        return pickle.load(open(pickle_dir + '/all_input.p','rb'))


if __name__ == '__main__':
    solar_array = SolarData(pickle=False)
    print solar_array.data[0].shape
