# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:25:57 2020
@author: combitech
"""
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model

#use this. does only work with 1 dense layer
def set_weights_from_model(source_model, destination_model):
    source_model = remove_all_dense(source_model)   
    last_layer = destination_model.layers[-1]
    destination_model = remove_all_dense(destination_model)
    i = 0
    for layer in source_model.layers:
        destination_model.layers[i].set_weights(layer.get_weights())
        i += 1
    #destination_model.set_weights(source_model.get_weights())
    return Model(inputs = destination_model.layers[0].output, outputs = last_layer.output)

def freeze_all_conv_layers(model):
    for layer in model.layers:    
        if not isinstance(layer, Flatten):
            layer.trainable = False
        else:
            break
    return model

def freeze_half_model(model):
    tot_layers = len(model.layers)
    limit = tot_layers / 2
    counter = 0
    print("limit: " + str(limit))
    for layer in model.layers:
        counter += 1
        if not isinstance(layer, Flatten):
            if counter < limit:
                layer.trainable = False
        print(str(counter) + ": " + str(layer.trainable))
    return model
    
def remove_last_layer(model):
    return Model(inputs = model.layers[0].output, outputs = model.layers[-2].output) #deletes the last layer.

def remove_all_dense(model):
    new_last_layer_index = -1
    for i in reversed(range(len(model.layers))):    
        if not isinstance(model.layers[i], Flatten):
            new_last_layer_index -= 1
        else:
            break
    return Model(inputs = model.layers[0].output, outputs = model.layers[new_last_layer_index].output) #deletes the last layer.
