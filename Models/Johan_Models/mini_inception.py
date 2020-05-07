# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:24:36 2020
@author: combitech
"""

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.models import Model

    
def common_layer(X):
    X = layers.BatchNormalization()(X)
    return layers.LeakyReLU()(X)

def mini_inception_module(X, strides):
    l0 = Conv2D(32,(1,1), padding='same')(X)
    l0 = common_layer(l0)
    l0 = Conv2D(64, (3,3), strides=strides, padding='same')(l0)
    l0 = common_layer(l0)
    
    l1 = Conv2D(32,(1,1), padding='same')(X)
    l1 = common_layer(l1)
    l1 = Conv2D(64, (5,5), strides=strides, padding='same')(l1)
    l1 = common_layer(l1)
    return layers.concatenate([l0,l1])


def init_model(in_dim, classes):
    input_layer = Input(shape = in_dim, dtype='float32', name='in')
   
    X = mini_inception_module(input_layer, 2)
    X = mini_inception_module(X,2)
    X = mini_inception_module(X,1)
    l11 = layers.AveragePooling2D((3,3), strides=2)(X)
    l12 = Flatten()(l11)
    #dense1 = Dense(100, activation='relu')(l6)
    dense2 = Dense(classes, activation='softmax')(l12)
    model = Model(inputs = input_layer, outputs = dense2, name="model") 
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save_model(model):
    experiment_name = "Transferlearning_mini_inception"
    location = '/home/johan/Experiment/Johan_Experiments/' + experiment_name + '/' + model.name + '.hdf5'
    model.save(location)