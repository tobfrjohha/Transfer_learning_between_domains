# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:10:41 2020

@author: combitech
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
import read_audio

#from tensorflow.keras.utils import plot_model
#tf.keras.utils.plot_model(inception_resnet, 'inception_resnet.png', show_shapes=True,show_layer_names =True)

def change_from_sequential(model):
    
    pre_layer = model.layers[0].layers[-1].output
    print(pre_layer)
    for layer in model.layers:
        if not isinstance(layer, Model):
            print(layer)
            pre_layer = layer (pre_layer)
    model = Model(inputs = model.layers[0].inputs, outputs = pre_layer)
    return model

base_model = tf.keras.applications.InceptionResNetV2(input_shape=(299,299,3),
                                                     include_top=True,
                                                     weights='imagenet')
base_model.summary()
img = 'Concert-Event-14'
printHMvariations(base_model, 'conv_7b_ac' , 'C:\\Users\\combitech\\Downloads\\'+img+'.jpg', (299, 299), "truck", "ResNet-v2")

##grad cam
model_mini_inception_clean = tf.keras.models.load_model("C:\\Users\\combitech\\Desktop\\Experiment\\Mini-Inception\\Mini-Inception, clean\\V22_model_06-0.11_best_val_model.hdf5")
model_mini_inception_clean.summary()
mini_inception_unfreezed = tf.keras.models.load_model("C:\\Users\\combitech\\Desktop\\Experiment\\Mini-Inception\\Mini-Inception, tf, unfreezed\\V2_model_29_110-0.2871_best_val_model.hdf5")
mini_inception_unfreezed.summary()
mini_inception_freezed = tf.keras.models.load_model("C:\\Users\\combitech\\Desktop\\Experiment\\Mini-Inception\\Mini-Inception, tf, freeze\\V26_model_11_41-0.3091_best_val_model.hdf5")
mini_inception_freezed.summary()

model_mini_inception_resnet_clean = tf.keras.models.load_model("C:\\Users\\combitech\\Desktop\\Experiment\\Mini-Inception-ResNet\\Mini-Inception-ResNet, clean\\V30_mini_inception_resnet_98-0.0902_best_val_model.hdf5")
model_mini_inception_resnet_clean.summary()
mini_inception_resnet_unfreezed = tf.keras.models.load_model("C:\\Users\\combitech\\Desktop\\Experiment\\Mini-Inception-ResNet\\Mini-Inception-ResNet, unfreezed\\V18_model_5_56-0.4697_best_val_model.hdf5")
mini_inception_resnet_unfreezed.summary()
mini_inception_resnet_freezed = tf.keras.models.load_model("C:\\Users\\combitech\\Desktop\\Experiment\\Mini-Inception-ResNet\\Mini-Inception-ResNet, tf, freeze\\V16_model_8_76-0.7750_best_val_model.hdf5")
mini_inception_resnet_freezed.summary()


inception_v3_clean = tf.keras.models.load_model("C:\\Users\\combitech\\Desktop\\Experiment\\Inception-V3\\Inception-V3_clean\\V17_sequential_17_best_val_model.hdf5")
inception_v3_clean = change_from_sequential(inception_v3_clean)
inception_v3_clean.summary()

inception_resnet_clean = tf.keras.models.load_model("C:\\Users\\combitech\\Desktop\\Experiment\\Inception-ResNet-V2\\Inception-ResNet-V2 clean\\V16_sequential_46_best_val_model.hdf5")
inception_resnet_clean = change_from_sequential(inception_resnet_clean)
inception_resnet_clean.summary()

mini_inception_resnet_unfreezed = tf.keras.models.load_model("C:\\Users\\combitech\\Desktop\\Experiment\\Mini-Inception-ResNet\\Mini-Inception-ResNet, unfreezed\\V18_model_5_56-0.4697_best_val_model.hdf5")
mini_inception_resnet_unfreezed.summary()


#count = 0
#for layer in model_mini_inception_resnet_clean.layers:
#    if isinstance(layer, Conv2D):
#        count += 1
#print (count)
#from tensorflow.keras.utils import plot_model
#tf.keras.utils.plot_model(model_mini_inception_resnet_clean, 'k.png', show_shapes=True,show_layer_names =True)



#clean
printHMvariations(model_mini_inception_clean, 'average_pooling2d_30' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Noise\\Noise3-121315.jpg', (128, 128), "Noise", "Mini-Inception_clean")
printHMvariations(model_mini_inception_resnet_clean, 'average_pooling2d_38' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Noise\\Noise3-121315.jpg', (128, 128), "Noise", "Mini-Inception-ResNet_clean")
printHMvariations(model_mini_inception_clean, 'average_pooling2d_30' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Racer\\Racer1-111.jpg', (128, 128), "Racer", "Mini-Inception_clean")
printHMvariations(model_mini_inception_resnet_clean, 'concatenate_116' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Racer\\Racer1-111.jpg', (128, 128), "Racer", "Mini-Inception-ResNet_clean")

#unfreezed
printHMvariations(mini_inception_unfreezed, 'average_pooling2d_109' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Noise\\Noise2-161149.jpg', (128, 128), "Noise", "Mini-Inception_unfreezed")
printHMvariations(mini_inception_resnet_unfreezed, 'concatenate_116' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Noise\\Noise2-161149.jpg', (128, 128), "Noise", "Mini-Inception-ResNet_unfreezed")
printHMvariations(mini_inception_unfreezed, 'average_pooling2d_109' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Racer\\Racer1-100.jpg', (128, 128), "Racer", "Mini-Inception_unfreezed")
printHMvariations(mini_inception_resnet_unfreezed, 'concatenate_116' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Racer\\Racer1-111.jpg', (128, 128), "Racer", "Mini-Inception-ResNet_unfreezed")

printHMvariations(mini_inception_freezed, 'average_pooling2d_2' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Noise\\Noise3-121315.jpg', (128, 128), "Noise", "Mini-Inception_freezed")
printHMvariations(mini_inception_resnet_freezed, 'concatenate_26' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Noise\\Noise3-121315.jpg', (128, 128), "Noise", "Mini-Inception-ResNet_freezed")
printHMvariations(mini_inception_freezed, 'average_pooling2d_2' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Racer\\Racer1-111.jpg', (128, 128), "Racer", "Mini-Inception_freezed")
printHMvariations(mini_inception_resnet_freezed, 'concatenate_26' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Racer\\Racer1-111.jpg', (128, 128), "Racer", "Mini-Inception-ResNet_freezed")


printHMvariations(mini_inception_freezed, 'average_pooling2d_2' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Spy_Cam\\Spy_Cam1-31048.jpg', (128, 128), "Spy", "Mini-Inception_freezed")


#printHMvariations(inception_v3_clean, 'conv2d_1638' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Noise\\Noise3-121315.jpg', (128, 128), "Noise")
#printHMvariations(inception_resnet_clean, 'conv2d_6064' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Noise\\Noise3-121315.jpg', (128, 128), "Noise")





printHMvariations(model_mini_inception, 'average_pooling2d_23' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Racer\\Racer1-100.jpg', (128, 128), "Racer")
printHMvariations(model_mini_inception_resnet, 'average_pooling2d_93' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Racer\\Racer1-100.jpg', (128, 128), "Racer")
#printHMvariations(inception_v3, 'mixed10' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Racer\\Racer1-100.jpg', (128, 128), "Racer")
#printHMvariations(inception_resnet, 'conv2d_6031' , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Racer\\Racer1-100.jpg', (128, 128), "Racer")







printHMvariations(model_trans, layers , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Racer\\Racer1-100.jpg', (128, 128), "Racer")
printHMvariations(model_trans, layers , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Spy_Cam\\Spy_Cam1-31048.jpg',  (128, 128), "Spy_Cam")
printHMvariations(model_trans, layers, 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Sub\\Red_Hat1-20024.jpg', (128, 128), "Sub")
printHMvariations(model_trans, layers , 'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Tugboat\\Tugboat1-20082.jpg', (128, 128), "Tugboat")
