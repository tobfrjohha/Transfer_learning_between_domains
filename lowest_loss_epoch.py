#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:15:02 2020

@author: johan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 07:59:04 2020

@author: johan
"""
# Imports
import csv
import os, shutil
from Dataset.datasets import boats_dataset
from Dataset.datasets import cifar_10_dataset
from tensorflow.keras.models import load_model
from tensorflow import keras

# user refers to the currently active user of the computer
user = "johan"

def find_min_loss_epoch(evaluator_name, experiment_name):
    #evaluator_name - the one responsible for the experiment
    #experiment_name - name of the current experiment
    #dataset - the dataset which is used. Returns train_x,train_y,test_x,test_y
    #multiple = True -> When multiple models exist in the same folder. Finds the best one and evaluates
    #multiple = False -> When single model exist in the folders
    
    evaluator = evaluator_name
    experiment = experiment_name

    # Copying best model of each version over to the evaluation folder
    saved_folder_name = os.path.join("/home/"+user+"/Experiment/" + evaluator + "_Experiments/" + experiment + "/saved_models/")
    os.chdir(saved_folder_name)                        
    header_exist = False
    # Header has not been set in the csv file
    for i in range(30):
        # Name of the folder containing the model(s)
        folder_name = os.path.join("/home/"+user+"/Experiment/" + evaluator + "_Experiments/" + experiment + "/saved_models/V"+str(i+1)+"_/")
        print("checking folder: " + str(folder_name))
        eval_files = os.listdir(folder_name)
        # If multiple models exist of the current version
        print("version: " + str(i+1))
        # Initial values when looking for the best model
        lowest_val = 99999.0
        best_epoch = -1
        val_acc = 0
        
        # Searching for the best model in the folder
        for files in eval_files:
            count = 0
            # Ensures that only model files are examined
            if ("training" in files):
                print("Printing to csv")
                #Writing the results to a csv file
                csv_file = open(folder_name + files, 'r')
                with csv_file as eval_file:
                        for line in csv_file.readlines():
                            if (count > 0):
                                array=line.split(';')
                                epoch=int(array[0]) + 1
                                val_loss=float(array[4])
                                
                                if (val_loss < lowest_val):
                                    lowest_val = val_loss 
                                    best_epoch = epoch 
                                    val_acc = float(array[3])
                            count+=1
                        print(str(lowest_val) + " at epoch: " + str(best_epoch) + ", with acc: " + str(val_acc))
                        
        csv_file = open(saved_folder_name + "_best_epoch.csv", 'a')
        with csv_file as eval_file:
            if (not header_exist):
                metrics = ['version', 'epoch', 'val_loss', 'val_accuracy']
                writer = csv.DictWriter(eval_file, delimiter=";", fieldnames=metrics) 
                writer.writeheader()
                writer.writerow({'version' : 'V_'+str(i+1), 'epoch' : best_epoch, 'val_loss' : lowest_val, 'val_accuracy' : val_acc})
                header_exist = True
            else:
                metrics = ['version', 'epoch', 'val_loss', 'val_accuracy']
                writer = csv.DictWriter(eval_file, delimiter=";", fieldnames=metrics)
                writer.writerow({'version' : 'V_'+str(i+1), 'epoch' : best_epoch, 'val_loss' : lowest_val, 'val_accuracy' : val_acc})
                        
            keras.backend.clear_session()
                        
#find_min_loss_epoch("Johan", "boats_mini_inception_tf_freeze")
#find_min_loss_epoch("Tobias", "boats_mini_inception_resnet")

#find_min_loss_epoch("Johan", "boats_mini_inception_tf_freeze_v2_new")
#find_min_loss_epoch("Johan", "boats_mini_inception_tf_new")
            
#find_min_loss_epoch("Tobias", "boats_mini_inception_resnet_tf_freeze_V2_new")
#find_min_loss_epoch("Tobias", "boats_mini_inception_resnet_tf_new")
            
#find_min_loss_epoch("Tobias", "boats_inception_resnet_v2_TF_HT")
#find_min_loss_epoch("Tobias", "boats_mini_inception_resnet_tf_HF_new")

#find_min_loss_epoch("Johan", "boats_inception_v3_TF_HT")