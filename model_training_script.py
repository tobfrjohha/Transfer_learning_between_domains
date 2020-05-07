# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 00:16:49 2020
@author: Tobias
"""

from Dataset.datasets import boats_dataset
from Dataset.datasets import cifar_10_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import csv
import os, shutil


def choose_ds(name):
    if name == "boats":
        tr_x,tr_y,te_x,te_y = boats_dataset()
    elif name == "cifar10":
        tr_x,tr_y,te_x,te_y = cifar_10_dataset()
    else:
        "invalid name"
        return;
    return tr_x, tr_y, te_x, te_y

def show(history):
    val_acc = history['val_accuracy']
    val_loss = history['val_loss']
    acc = history['accuracy']
    loss = history['loss']
    epochs = range(len(acc))
    ax = plt.subplot(111)
    ax.plot(epochs, acc, 'b', label='Training acc')
    ax.plot(epochs, loss, 'c', label='Training loss')
    ax.plot(epochs, val_acc, 'r', label='Validation acc')
    ax.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.ylim(top=1.5, bottom=0.0)
    plt.title("Training History")
    ax.legend()

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=20.0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.2,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.2,
        shear_range=0.15,  # set range for random shear
        zoom_range=0.15,  # set range for random zoom
        channel_shift_range=0.15,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        #vertical_flip=True,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).


def train_model_augmented(new_model, dataset, iteration, experimentor, experiment_name):
    model_location_on_drive = '/home/johan/Experiment/' + experimentor + '_Experiments/' + experiment_name + '/'
    
    train_x, train_y, test_x, test_y = choose_ds(dataset)

    version = "V" + str(iteration+1) + "_"
    model = new_model
    model_name = model.name
    IMG_SIZE=model.input.shape[1]
    
    saved_folder_name = folder_name = os.path.join(model_location_on_drive + 'saved_models/')
    if (not os.path.exists(saved_folder_name)):
        os.mkdir        (saved_folder_name)
    
    folder_name = os.path.join(model_location_on_drive + 'saved_models/' + version + '/')

    if (not os.path.exists(folder_name)):
        os.mkdir(folder_name)
        
    train_x = np.array(train_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    train_y = np.array(train_y)
    test_x = np.array(test_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    test_y = np.array(test_y)
    
    checkpoint_val_loss = ModelCheckpoint(folder_name + version + model_name + "_{epoch:02d}-{val_loss:.2f}_best_val_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    datagen.fit(train_x)

    model.fit_generator(datagen.flow(train_x, train_y,
                batch_size=32),
                epochs=300,
                validation_data=(test_x, test_y),
                callbacks=[checkpoint_val_loss])
    
    if (not os.path.exists(saved_folder_name + "/evaluation")):
        os.mkdir(saved_folder_name + "/evaluation")
    
    os.chdir(saved_folder_name)
    eval_files = os.listdir(folder_name)
    
    # Initial values when looking for the best model
    best_model = None
    lowest_val = 99999
    best_epoch = -1
    
    # Searching for the best model in the folder
    for files in eval_files:
        substring = files
        if ((not "training" in files) and (not "evaluation" in files)):
            substring = files.split(".")
            get_epoch = substring[0].split("-")
            get_epoch = get_epoch[0].split("_")
            print(get_epoch[len(get_epoch) - 1])
            epoch = int(get_epoch[len(get_epoch) - 1])
            print(substring)
            substring = substring[1].split("_")
            val = int(substring[0])
            if (val <= lowest_val):
                if (epoch > best_epoch):
                    lowest_val = val
                    best_model = files
                    best_epoch = epoch
                                        
    # The best model is copied over the the evaluation folder
    print("copying: " + best_model)
    shutil.copy(folder_name + best_model, saved_folder_name + "/evaluation")

def train_model(new_model, dataset, iteration, experimentor, experiment_name):
    model_location_on_drive = '/home/johan/Experiment/' + experimentor + '_Experiments/' + experiment_name + '/'
    
    train_x, train_y, test_x, test_y = choose_ds(dataset)

    version = "V" + str(iteration+1) + "_"
    model = new_model
    model_name = model.name
    IMG_SIZE=model.input.shape[1]
    
    # Create the saved_models folder
    saved_folder_name = folder_name = os.path.join(model_location_on_drive + 'saved_models/')
    if (not os.path.exists(saved_folder_name)):
        os.mkdir(saved_folder_name)
    
    # Create the version folder
    folder_name = os.path.join(model_location_on_drive + 'saved_models/' + version + '/')
    if (not os.path.exists(folder_name)):
        os.mkdir(folder_name)
        
    train_x = np.array(train_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    train_y = np.array(train_y)
    test_x = np.array(test_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    test_y = np.array(test_y)
    
    csv_logger = CSVLogger(folder_name + version + str(model_name) + '_training.csv', separator=';')
    checkpoint_val_loss = ModelCheckpoint(folder_name + version + model_name + "_{epoch:02d}-{val_loss:.4f}_best_val_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=125, verbose=1)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    model.fit(train_x, train_y, batch_size=32, validation_split=0.10, epochs = 1000, callbacks=[checkpoint_val_loss, csv_logger, early_stopping])
    
    # Create the evaluation foldermodel = load_model("/home/johan/Experiment/Tobias_Experiments/Transferlearning_mini_inception_resnet/saved_models/V1_/V1_mini_inception_resnet_45-0.54_best_val_model.hdf5")


    if (not os.path.exists(saved_folder_name + "/evaluation")):
        os.mkdir(saved_folder_name + "/evaluation")
    
    os.chdir(saved_folder_name)
    eval_files = os.listdir(folder_name)
    
    # Initial values when looking for the best model
    best_model = None
    lowest_val = 99999
    best_epoch = -1
    
    # Searching for the best model in the folder
    for files in eval_files:
        substring = files
        if ((not "training" in files) and (not "evaluation" in files)):
            substring = files.split(".")
            get_epoch = substring[0].split("-")
            get_epoch = get_epoch[0].split("_")
            print(get_epoch[len(get_epoch) - 1])
            epoch = int(get_epoch[len(get_epoch) - 1])
            print(substring)
            substring = substring[1].split("_")
            val = int(substring[0])
            if (val <= lowest_val):
                if (epoch > best_epoch):
                    lowest_val = val
                    best_model = files
                    best_epoch = epoch
    
    # The best model is copied over the the evaluation folder
    print("copying: " + best_model)
    shutil.copy(folder_name + best_model, saved_folder_name + "/evaluation")
        
    #for file in folder_name:
    
    print("MODEL EVALUATION:")
    eval_metrics = model.evaluate(test_x, test_y, batch_size=32, verbose=2)
    file = open(folder_name + version + str(model_name) +'_evaluation.csv', 'a')
    with file as eval_file:
        metrics = ['loss', 'accuracy']
        writer = csv.DictWriter(eval_file, delimiter=";", fieldnames=metrics) 
        writer.writeheader()
        writer.writerow({'loss' : eval_metrics[0], 'accuracy' : eval_metrics[1]})