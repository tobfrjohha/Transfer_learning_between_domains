import csv
import os, shutil
from Dataset.datasets import boats_dataset
from Dataset.datasets import cifar_10_dataset
from tensorflow.keras.models import load_model
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas, xlsxwriter
from array import *

user = "johan"

def write_confusion_matrix_to_csv(evaluator_name, experiment_name, eval_folder, saved_models_folder, multiple = False):
    train_x, train_y, test_x, test_y = boats_dataset()
    
    header_exist = False
    for i in range(30):
        # Name of the folder containing the model(s)
        folder_name = os.path.join("/home/"+user+"/Experiment/" + evaluator_name + "_Experiments/" + experiment_name + "/saved_models/V"+str(i+1)+"_/")
        #print("checking folder: " + str(folder_name))
        eval_files = os.listdir(folder_name)
        
        # If multiple models exist of the current version
        print("version: " + str(i+1))
        # Initial values when looking for the best model
        best_model = None
        lowest_val = 99999
        best_epoch = -1
        if multiple:
            # Searching for the best model in the folder
            for files in eval_files:
                print(files)
                substring = files
                # Ensures that only model files are examined
                if ((not "training" in files) and (not "evaluation" in files)):
                    substring = files.split(".")
                    get_epoch = substring[0].split("-")
                    get_epoch = get_epoch[0].split("_")
                    #print(get_epoch[len(get_epoch) - 1])
                    print(get_epoch)
                    epoch = int(get_epoch[len(get_epoch) - 1])
                    print(epoch)
                    #print(substring)
                    substring = substring[1].split("_")
                    val = int(substring[0])
                    if (val <= lowest_val):
                        if (epoch > best_epoch):
                            lowest_val = val
                            best_model = files
                            best_epoch = epoch
            print("Loading model: " + str(best_model))
            # Loads the best model that was found
            model = load_model(folder_name + best_model)
            print('---------------------------------------------')
            y_pred = model.predict(test_x)
            y_pred = y_pred.argmax(axis=1)
            target_names = ['class 0(noise)', 'class 1(racer)', 'class 2(spy_cam)', 'class 3(sub)', 'class 4(tugboat)']
            #print('---------------------------------------------')
            report = classification_report(test_y,y_pred, output_dict=True, target_names=target_names)
            print(type(report))
            df = pandas.DataFrame(report).transpose()
            
            #print('---------------------------------------------')
            matrix = confusion_matrix(test_y, y_pred)
            print(type(matrix))
            print (df)

            #print(matrix[0])
            
            csv_file = open(eval_folder + "/_CM_statistics_" + experiment_name + ".csv", 'a')
            with csv_file as stats_file:
                if (not header_exist):
                    metrics = target_names
                    metrics.append('version')
                    writer = csv.DictWriter(stats_file, delimiter=";", fieldnames=metrics) 
                    writer.writeheader()
                    writer.writerow({'class 0(noise)' : matrix[0], 'class 1(racer)' : matrix[1], 'class 2(spy_cam)' : matrix[2], 'class 3(sub)' : matrix[3], 'class 4(tugboat)' : matrix[4], 'version' : str(i+ 1)})
                    header_exist = True
                else:
                    metrics = target_names
                    metrics.append('version')
                    writer = csv.DictWriter(stats_file, delimiter=";", fieldnames=metrics) 
                    writer.writerow({'class 0(noise)' : matrix[0], 'class 1(racer)' : matrix[1], 'class 2(spy_cam)' : matrix[2], 'class 3(sub)' : matrix[3], 'class 4(tugboat)' : matrix[4], 'version' : str(i + 1)})
         
            #csv_file = open(eval_folder + "/_REPORT_statistics_" + experiment_name + ".csv", 'a')
            #workbook = xlsxwriter.Workbook(eval_folder + "/_REPORT_statistics_" + experiment_name + ".xlsx")
            #worksheet = workbook.add_worksheet()
            
            report_folder =  os.path.join(eval_folder + "/reports")
            if (not os.path.exists(report_folder)):
                os.mkdir(report_folder)
            writer = pandas.ExcelWriter(eval_folder + "/reports/V"+str(i+ 1)+"_REPORT_statistics_" + experiment_name + '.xlsx', engine='xlsxwriter')
            df.to_excel(writer, sheet_name = "report")
            writer.save()
            
            del model
            keras.backend.clear_session()

def write_statistics_to_csv(evaluator_name, experiment_name, eval_folder, saved_models_folder, multiple = False):
    train_x, train_y, test_x, test_y = boats_dataset()
    
    header_exist = False
    for i in range(30):
        print(experiment_name)
        print(experiment_name)
        print(experiment_name)
        # Name of the folder containing the model(s)
        folder_name = os.path.join("/home/"+user+"/Experiment/" + evaluator_name + "_Experiments/" + experiment_name + "/saved_models/V"+str(i+1)+"_/")
        #print("checking folder: " + str(folder_name))
        eval_files = os.listdir(folder_name)
        # If multiple models exist of the current version
        print("version: " + str(i+1))
        # Initial values when looking for the best model
        best_model = None
        lowest_val = 99999
        best_epoch = -1
        if multiple:
            # Searching for the best model in the folder
            for files in eval_files:
                substring = files
                # Ensures that only model files are examined
                if ((not "training" in files) and (not "evaluation" in files)):
                    substring = files.split(".")
                    get_epoch = substring[0].split("-")
                    get_epoch = get_epoch[0].split("_")
                    #print(get_epoch[len(get_epoch) - 1])
                    epoch = int(get_epoch[len(get_epoch) - 1])
                    #print(substring)
                    substring = substring[1].split("_")
                    val = int(substring[0])
                    if (val <= lowest_val):
                        if (epoch > best_epoch):
                            lowest_val = val
                            best_model = files
                            best_epoch = epoch
            print("Loading model: " + str(best_model))
            # Loads the best model that was found
            model = load_model(folder_name + best_model)
            print('---------------------------------------------')
            y_pred = model.predict(test_x)
            y_pred = y_pred.argmax(axis=1)
            target_names = ['','(precision)', '(recall)', '(f1-score)', '(support)']
            #print('---------------------------------------------')
            report = classification_report(test_y,y_pred, output_dict=True, target_names=target_names)
            print(type(report))
            df = pandas.DataFrame(report).transpose()
            
            #print('---------------------------------------------')
            matrix = confusion_matrix(test_y, y_pred)
            print(matrix)
           # for col in df.columns.values:
            #    print(col)
            vals = []
            cols = []
            
            row = 0
            for col in df:
                cols.append(col)
            print(cols, end='---')
            for values in df.values:
                for value in values:
                    vals.append([])
                    vals[row].append(value)

                row += 1
            print(vals[0][0])
            print('---')
            print(df)
            
            
            del model
            keras.backend.clear_session()
    
def statistic_analysis(evaluator_name, experiment_name, multiple = False, cm=False):
    #evaluator_name - the one responsible for the experiment
    #experiment_name - name of the current experiment
    #dataset - the dataset which is used. Returns train_x,train_y,test_x,test_y
    #multiple = True -> When multiple models exist in the same folder. Finds the best one and evaluates
    #multiple = False -> When single model exist in the folders

    # Set the evaluation folder
    evaluation_folder =  os.path.join("/home/"+user+"/Experiment/" + evaluator_name + "_Experiments/" + experiment_name + "/saved_models/evaluation")
    if (not os.path.exists(evaluation_folder)):
        os.mkdir(evaluation_folder)
    
    # Copying best model of each version over to the evaluation folder
    saved_folder_name = os.path.join("/home/"+user+"/Experiment/" + evaluator_name + "_Experiments/" + experiment_name + "/saved_models/")
    os.chdir(saved_folder_name)
    if (cm==True):
        write_confusion_matrix_to_csv(evaluator_name, experiment_name, evaluation_folder, saved_folder_name, multiple)
    else:
        write_statistics_to_csv(evaluator_name, experiment_name, evaluation_folder, saved_folder_name, multiple)

#predict_with_single_model("Tobias", "Mini-Inception-ResNet, RC-Boats, tf, unfreezed", boats_dataset(), "26", "V26_model_77_12-0.1248_best_val_model.hdf5")
#statistic_analysis("Tobias", "Mini-Inception-ResNet, RC-Boats, tf, unfreezed", multiple = True)
#statistic_analysis("Tobias", "Mini-Inception-ResNet, RC-Boats, tf, half freezed", multiple = True, cm = True)
statistic_analysis("Tobias", "Mini-Inception-ResNet, RC-Boats, tf, half freezed", multiple = True, cm = False)
