import xlsxwriter
import random
import csv
import os
import numpy as np

user = "johan"
def generate_graph_from_csv(experimentor, experiment, acc_graphs = False, length = 25):
    # experimentor -> who has responsibility for the experiment
    # experiment -> the target experiment
    # acc_graphs = True -> produce graphs for accuracy
    # acc_graphs = false -> produce graphs for loss
    # length -> how many epochs that are being looked at, from 0 to length.
    if (acc_graphs == True):
        first_index = 1
        val_index = 3
        param = "Accuracy"
        graph_type = "ACC"
    else:
        first_index = 2
        val_index = 4
        param = "Loss"
        graph_type = "LOSS"
    
    _data = [] # Data location inside excel
    _val_data = []
    data_start_loc = [0, 0] # xlsxwriter rquires list, no tuple
    data_start_loc_val = [0, 1]
    
    location = os.path.join("/home/"+user+"/Experiment/" + experimentor + "_Experiments/" + experiment + "/")
    files = os.listdir(location)
    for file in files:
       # print(file)
        if ("_average_results.csv" in file):
            
            with open(location + file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=";")
                row_count = 0
                for row in csv_reader:
                    if (row_count > 0):
                        _data.append(float(row[first_index]))
                        _val_data.append(float(row[val_index]))
                    else: 
                        _data.append(row[first_index])
                        _val_data.append(row[val_index])
                    row_count += 1

    data_end_loc = [data_start_loc[0] + length, 0]
    data_end_loc_val = [data_start_loc[0] + length, 1]
    
    workbook = xlsxwriter.Workbook('Graphs/' + str(param) + '/graph_' + str(graph_type) + '_' + experiment + '.xlsx')
    
    # Charts are independent of worksheets
    chart = workbook.add_chart({'type': 'line'})
    chart.set_y_axis({'name': str(param)})
    chart.set_x_axis({'name': 'Epochs'})
    chart.set_title({'name': str(experiment) + "\nMean training curve"})
    
    worksheet = workbook.add_worksheet()
    
    # A chart requires data to reference data inside excel
    worksheet.write_column(*data_start_loc, data=_data)
    
    worksheet.write_column(*data_start_loc_val, data=_val_data)
    # The chart needs to explicitly reference data
    data_start_loc = [1, 0]
    data_start_loc_val = [1, 1]
    
    chart.add_series({
        'values': [worksheet.name] + data_start_loc + data_end_loc,
        'name': str(param),
    })
    
    chart.add_series({
        'values': [worksheet.name] + data_start_loc_val + data_end_loc_val,
        'name': "Val_" + str(param),
    })
    worksheet.insert_chart('C1', chart)
    
    workbook.close()  # Write to file

def make_graphs(experimentor, acc_graphs = False):
    experiments_location = os.listdir("/home/"+user+"/Experiment/" + experimentor + "_Experiments/")
    for experiment in experiments_location:
        print(experiment)
        if(not "Transferlearning_" in experiment):
            generate_graph_from_csv(str(experimentor), str(experiment), acc_graphs)

make_graphs("Tobias", acc_graphs = True)
make_graphs("Johan", acc_graphs = True)
make_graphs("Tobias", acc_graphs = False)
make_graphs("Johan", acc_graphs = False)
