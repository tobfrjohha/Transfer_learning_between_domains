import csv
import os, shutil
import numpy as np

user = "johan"

def printAvgResultsToCSV(experimentor, experiment):
    header_exist = False
    location = "/home/"+user+"/Experiment/" + experimentor + "_Experiments/" + experiment + "/"
    csv_avg_file = open(location + "/_average_results.csv", 'a')
    results = [0.0, 0.0, 0.0, 0.0]
    epoch_arr = []
    
    versions_with_curr_epoch = []   #keeping track of models with the current epoch. Important for average results
    #loop through all versions
    for version in range(1, 31):
        folder = os.path.join(location + "saved_models/V" + str(version) + "_/")
        eval_files = os.listdir(folder)
        #loop find the _training.csv file in version folder
        for files in eval_files:
            epoch_counter = 0
            if ("_training.csv" in files):
                #Open file
                with open(folder + files) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=";")
                    line_count = 0
                    #loop over every row in the file
                    for row in csv_reader:
                        #first row is the header
                        if (line_count == 0) :
                            line_count += 1
                        else:
                            if (len(epoch_arr) < (epoch_counter + 1)):
                                epoch_arr.append(results)
                            if (len(versions_with_curr_epoch) < (epoch_counter +1)):
                                versions_with_curr_epoch.append(0)
                            versions_with_curr_epoch[epoch_counter] = versions_with_curr_epoch[epoch_counter] + int(1)
                            #add the new values to the array with the results for every epoch
                            epoch_arr[epoch_counter] = [
                                    epoch_arr[epoch_counter][0] + float(row[1]), 
                                    epoch_arr[epoch_counter][1] + float(row[2]), 
                                    epoch_arr[epoch_counter][2] + float(row[3]), 
                                    epoch_arr[epoch_counter][3] + float(row[4])
                                    ]

                            epoch_counter += 1
                            line_count += 1
        
        #if we have reached the final version
        if (version == 30):
            counter = 0
            line_count = 0
            for i in epoch_arr:
                #accuracy
                #print("Epoch: " + str(counter + 1) + " is used by " + str(versions_with_curr_epoch[counter]) + " versions")
                epoch_arr[counter][0] = float(epoch_arr[counter][0])/versions_with_curr_epoch[counter]
                #loss
                epoch_arr[counter][1] = float(epoch_arr[counter][1])/versions_with_curr_epoch[counter]
                #val_acc
                epoch_arr[counter][2] = float(epoch_arr[counter][2])/versions_with_curr_epoch[counter]
                #val_loss
                epoch_arr[counter][3] = float(epoch_arr[counter][3])/versions_with_curr_epoch[counter]

                metrics = ['Epoch','Accuracy','Loss', 'Val_Accuracy', 'Val_Loss']
                if (header_exist == False):
                    writer = csv.DictWriter(csv_avg_file, delimiter=";", fieldnames=metrics) 
                    writer.writeheader()
                    header_exist = True
                
                epoch = counter + 1
                writer = csv.DictWriter(csv_avg_file, delimiter=";", fieldnames=metrics) 
                writer.writerow({'Epoch': epoch, 'Accuracy': epoch_arr[counter][0], 'Loss' : epoch_arr[counter][1], 'Val_Accuracy' : epoch_arr[counter][2], 'Val_Loss' : epoch_arr[counter][3]})
                line_count += 1
                counter += 1   
        #print(accuracy)

#printAvgResultsToCSV("Tobias", "boats_mini_inception_resnet_TEST")
#printAvgResultsToCSV("Johan", "boats_inception_v3")

def save_for_all_experiments(experimentor):
    experiments_location = os.listdir("/home/"+user+"/Experiment/" + experimentor + "_Experiments/")
    for experiment in experiments_location:
        if(not "Transferlearning_" in experiment):
            printAvgResultsToCSV(str(experimentor), str(experiment))
    
#save_for_all_experiments("Tobias")
#save_for_all_experiments("Johan")
