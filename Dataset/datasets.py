#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:06:34 2020

@author: johan
"""
import numpy as np
import pickle

def cifar_10_dataset():
    from tensorflow.keras.datasets import cifar10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x /= 255
    test_x /= 255
    return train_x, train_y, test_x, test_y

def boats_dataset():
    location = '/home/johan/Experiment/Dataset/Boats/'
    #TRAINING DATA
    pickle_in = open(location + "training_set_X.pickle", "rb")
    train_x = pickle.load(pickle_in)
    #Answers to training data
    pickle_in = open(location + "training_set_y.pickle", "rb")
    train_y = pickle.load(pickle_in)
    
    #TEST DATA
    pickle_in = open(location + "test_set_X.pickle", "rb")
    test_x = pickle.load(pickle_in)
    #Answers to training data
    pickle_in = open(location + "test_set_y.pickle", "rb")
    test_y = pickle.load(pickle_in)
    
    #Normalize data -- scale the data
    train_x = train_x/255.0
    test_x = test_x/255.0
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    return train_x, train_y, test_x, test_y