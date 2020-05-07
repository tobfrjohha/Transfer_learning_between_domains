# Imports
import csv
import os, shutil
from Dataset.datasets import boats_dataset
from Dataset.datasets import cifar_10_dataset
from tensorflow.keras.models import load_model
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# user refers to the currently active user of the computer
user = "johan"

def evaluate_single_model(evaluator_name, experiment_name, dataset, version, model_name):
    model = load_model("/home/" +user+ "/Experiment/" + evaluator_name + "_Experiments/" +experiment_name+ "/saved_models/V" +version+ "_/" +model_name)
    train_x, train_y, test_x, test_y = dataset
    model.evaluate(test_x, test_y, batch_size=32, verbose=2)

def evaluate_models_to_csv(evaluator_name, experiment_name, dataset, multiple = False):
    #evaluator_name - the one responsible for the experiment
    #experiment_name - name of the current experiment
    #dataset - the dataset which is used. Returns train_x,train_y,test_x,test_y
    #multiple = True -> When multiple models exist in the same folder. Finds the best one and evaluates
    #multiple = False -> When single model exist in the folders
    
    train_x, train_y, test_x, test_y = dataset
    evaluator = evaluator_name
    experiment = experiment_name
    # Set the evaluation folder
    evaluation_folder =  os.path.join("/home/"+user+"/Experiment/" + evaluator + "_Experiments/" + experiment + "/saved_models/evaluation")
    if (not os.path.exists(evaluation_folder)):
        os.mkdir(evaluation_folder)
    
    # Copying best model of each version over to the evaluation folder
    saved_folder_name = os.path.join("/home/"+user+"/Experiment/" + evaluator + "_Experiments/" + experiment + "/saved_models/")
    os.chdir(saved_folder_name)
    
    # Header has not been set in the csv file
    header_exist = False
    for i in range(30):
        # Name of the folder containing the model(s)
        folder_name = os.path.join("/home/"+user+"/Experiment/" + evaluator + "_Experiments/" + experiment + "/saved_models/V"+str(i+1)+"_/")
        print("checking folder: " + str(folder_name))
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
            print("Loading model: " + str(best_model))
            # Loads the best model that was found
            model = load_model(folder_name + best_model)
            print("Model " + str(best_model)+ " is loaded")

            print("Evaluation...")
            # Evaluates the best model that was found
            eval_metrics = model.evaluate(test_x, test_y, batch_size=32, verbose=2)
            
            print("Printing to csv")
            #Writing the results to a csv file
            csv_file = open(evaluation_folder + "/_evaluation.csv", 'a')
            with csv_file as eval_file:
                if (not header_exist):
                    metrics = ['loss', 'accuracy', 'model']
                    writer = csv.DictWriter(eval_file, delimiter=";", fieldnames=metrics) 
                    writer.writeheader()
                    writer.writerow({'loss' : eval_metrics[0], 'accuracy' : eval_metrics[1], 'model' : model.name})
                    header_exist = True
                else:
                    metrics = ['loss', 'accuracy', 'model']
                    writer = csv.DictWriter(eval_file, delimiter=";", fieldnames=metrics) 
                    writer.writerow({'loss' : eval_metrics[0], 'accuracy' : eval_metrics[1], 'model' : model.name})
            print("Evaluation complete...")
            # Memory cleaning
            del model
            keras.backend.clear_session()
        else:
            #If single model exist of the current model
            for files in eval_files:
                
                if ((not "training" in files) and (not "evaluation" in files)):
                    best_model = files
                    print("Loading model: " + str(best_model))
                    model = load_model(folder_name + best_model)
                    print("Model " + str(best_model)+ " is loaded")
                    
                    print("Evaluation...")
                    eval_metrics = model.evaluate(test_x, test_y, batch_size=32, verbose=2)
                    
                    print("Printing to csv")
                    csv_file = open(evaluation_folder + "/_evaluation.csv", 'a')
                    with csv_file as eval_file:
                        if (not header_exist):
                            metrics = ['loss', 'accuracy', 'model']
                            writer = csv.DictWriter(eval_file, delimiter=";", fieldnames=metrics) 
                            writer.writeheader()
                            writer.writerow({'loss' : eval_metrics[0], 'accuracy' : eval_metrics[1], 'model' : model.name})
                            header_exist = True
                        else:
                            metrics = ['loss', 'accuracy', 'model']
                            writer = csv.DictWriter(eval_file, delimiter=";", fieldnames=metrics) 
                            writer.writerow({'loss' : eval_metrics[0], 'accuracy' : eval_metrics[1], 'model' : model.name})
                    print("Evaluation complete...")
                    del model
                    keras.backend.clear_session()
        # The best model is copied over the the evaluation folder
        #print("copying: " + best_model)
        #shutil.copy(folder_name + best_model, saved_folder_name + "/evaluation")
        

# stored execution of evaluation history
#evaluate_models_to_csv("Johan", "boats_inception_v3", boats_dataset(), multiple=False)
#evaluate_models_to_csv("Johan", "boats_inception_v3_initialized_weights", boats_dataset(), multiple=False)
#evaluate_models_to_csv("Johan", "boats_inception_v3_TF", boats_dataset(), multiple=False)
#model = load_model("/home/johan/Experiment/Tobias_Experiments/boats_inception_resnet_v2/saved_models/V1_/V1_sequential_31_best_val_model.hdf5")
#evaluate_models_to_csv("Johan", "Transferlearning_mini_inception", cifar_10_dataset(), multiple=True)

#evaluate_models_to_csv("Tobias", "boats_inception_resnet_v2", boats_dataset(), multiple=False)
#evaluate_models_to_csv("Tobias", "boats_inception_resnet_v2_initialized_weights", boats_dataset(), multiple=False)
#evaluate_models_to_csv("Tobias", "boats_inception_resnet_v2_TF", boats_dataset(), multiple=False)
#evaluate_models_to_csv("Tobias", "Transferlearning_mini_inception_resnet", cifar_10_dataset(), multiple=True)
        
#evaluate_models_to_csv("Tobias", "boats_mini_inception_resnet", boats_dataset(), multiple=False)

#evaluate_single_model("Tobias", "Transferlearning_mini_inception_resnet", cifar_10_dataset(), "1", "V1_mini_inception_resnet_45-0.54_best_val_model.hdf5")
#evaluate_single_model("Johan", "Transferlearning_mini_inception", cifar_10_dataset(), "1", "V1_model_43-0.59_best_val_model.hdf5")

#evaluate_models_to_csv("Tobias", "boats_mini_inception_resnet_tf_HF", boats_dataset(), multiple=True)
#evaluate_models_to_csv("Tobias", "boats_inception_resnet_v2_TF_HT", boats_dataset(), multiple=False)

#evaluate_models_to_csv("Johan", "boats_mini_inception_tf_HF", boats_dataset(), multiple=True)
#evaluate_models_to_csv("Johan", "boats_inception_v3_TF_HT", boats_dataset(), multiple=False)
        
#evaluate_models_to_csv("Tobias", "boats_mini_inception_resnet_tf_new", boats_dataset(), multiple=True)
#evaluate_models_to_csv("Tobias", "boats_mini_inception_resnet_tf_HF_new", boats_dataset(), multiple=True)
#evaluate_models_to_csv("Tobias", "boats_mini_inception_resnet_tf_freeze_V2_new", boats_dataset(), multiple=True)

#evaluate_models_to_csv("Johan", "boats_mini_inception_tf_new", boats_dataset(), multiple=True)
#evaluate_models_to_csv("Johan", "boats_mini_inception_tf_HF_new", boats_dataset(), multiple=True)
#evaluate_models_to_csv("Johan", "boats_mini_inception_tf_freeze_v2_new", boats_dataset(), multiple=True)


