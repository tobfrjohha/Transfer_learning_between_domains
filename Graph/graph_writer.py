import xlsxwriter, os, csv, pandas
import numpy as np


def generate_architecture_mean_training_graph(measure, architecture, epochs, max_y_loss , isNot="-1"):
    if(measure == 'Accuracy'):
        training = 'Accuracy'
        validation = 'Val_Accuracy'
        location = 'C:/Users/combitech/Desktop/Experiment/Graphs/Accuracy/'
    elif(measure == 'Loss'):
        training = 'Loss'
        validation = 'Val_Loss'
        location = 'C:/Users/combitech/Desktop/Experiment/Graphs/Loss/'     
    
     
    
    workbook = xlsxwriter.Workbook(location + measure +'_'+ architecture +'.xlsx')
    worksheet = workbook.add_worksheet()
    i = 0
    
    chart = workbook.add_chart({'type': 'line'})
    
    if measure == 'Loss':
        chart.set_y_axis({'name': training, 'max' : max_y_loss})
    else:
        chart.set_y_axis({'name': training, 'max' : 1.0, 'min' : 0.9})
    chart.set_x_axis({'name': 'Epoch'})
    chart.set_title({'name': architecture})
        
    chart2 = workbook.add_chart({'type': 'line'})
    
    if measure == 'Loss':
        chart2.set_y_axis({'name': validation, 'max' : max_y_loss})
    else:
        chart2.set_y_axis({'name': validation, 'max' : 1.0, 'min' : 0.9})
    chart2.set_x_axis({'name': 'Epoch'})
    chart2.set_title({'name': architecture})
    
    order_of_configurations = ["","","",""]
    for file in os.listdir(location):   
        if architecture in file:          
            if not isNot in file:
                if "random weights" in file:
                    order_of_configurations[0] = file
                elif "unfreezed" in file:
                    order_of_configurations[1] = file  
                elif "half freezed" in file:
                    order_of_configurations[2] = file
                elif "tf, freezed" in file:
                    order_of_configurations[3] = file
                print(file)         
    for file in order_of_configurations:
        config = file.split("RC-Boats, ")[1].split('.')[0]
        file = pandas.read_excel(location + file, sep=';')
        data_start = [0,i*2]
        data_start2 = [0,1+i*2]        
        data_end = [data_start[0] + epochs,i*2]
        data_end2 = [data_start2[0] + epochs,i*2+1]
        i+=1
        if measure == 'Loss':
            worksheet.write_column(*data_start, data=file.Loss)
            worksheet.write_column(*data_start2, data=file.Val_Loss)
        elif measure == 'Accuracy':
            worksheet.write_column(*data_start, data=file.Accuracy)
            worksheet.write_column(*data_start2, data=file.Val_Accuracy)
        chart.add_series({
            'values': [worksheet.name] + data_start + data_end,
            'name': config,})
        chart2.add_series({
            'values': [worksheet.name] + data_start2 + data_end2,
            'name': config,})
        
    worksheet.insert_chart('I1', chart)  
    worksheet.insert_chart('I16', chart2)        
    workbook.close() 
        

generate_architecture_mean_training_graph('Loss', 'Inception-ResNet-V2')
generate_architecture_mean_training_graph('Accuracy', 'Inception-ResNet-V2')

generate_architecture_mean_training_graph('Loss', 'Inception-v3')
generate_architecture_mean_training_graph('Accuracy', 'Inception-v3')

generate_architecture_mean_training_graph('Loss', 'Mini-inception', 150, 0.3,'ResNet')
generate_architecture_mean_training_graph('Accuracy', 'Mini-inception', 'ResNet')

generate_architecture_mean_training_graph('Loss', 'Mini-Inception-ResNet')
generate_architecture_mean_training_graph('Accuracy', 'Mini-Inception-ResNet')




def generate_configuration_mean_training_graph(measure, configuration, isNot="-1"):
    if(measure == 'Accuracy'):
        training = 'Accuracy'
        validation = 'Val_Accuracy'
        location = 'C:/Users/combitech/Desktop/Experiment/Graphs/Accuracy/'
    elif(measure == 'Loss'):
        training = 'Loss'
        validation = 'Val_Loss'
        location = 'C:/Users/combitech/Desktop/Experiment/Graphs/Loss/'     
    
     
    
    workbook = xlsxwriter.Workbook(location + measure +'_'+ configuration +'.xlsx')
    worksheet = workbook.add_worksheet()
    i = 0
    
    chart = workbook.add_chart({'type': 'line'})
    
    if measure == 'Loss':
        chart.set_y_axis({'name': training, 'max' : 0.9})
    else:
        chart.set_y_axis({'name': training, 'max' : 1.0, 'min' : 0.9})
    chart.set_x_axis({'name': 'Epoch'})
    chart.set_title({'name': configuration})
        
    chart2 = workbook.add_chart({'type': 'line'})
    
    if measure == 'Loss':
        chart2.set_y_axis({'name': validation, 'max' : 0.9})
    else:
        chart2.set_y_axis({'name': validation, 'max' : 1.0, 'min' : 0.9})
    chart2.set_x_axis({'name': 'Epoch'})
    chart2.set_title({'name': configuration})
    
    order_of_architecture = ["","","",""]
    for file in os.listdir(location):   
        if configuration in file:          
            if not isNot in file:
                if "Inception-ResNet-V2" in file:
                    order_of_architecture[0] = file
                elif "Inception-v3" in file:
                    order_of_architecture[1] = file  
                elif "Mini-inception," in file:
                    order_of_architecture[2] = file
                elif "Mini-Inception-ResNet" in file:
                    order_of_architecture[3] = file
                #print(file)         
    for file in order_of_architecture:
        architecture = file.split('_')[2].split(',')[0]
        config = file.split("RC-Boats, ")[1].split('.')[0]
        file = pandas.read_excel(location + file, sep=';')
        data_start = [0,i*2]
        data_start2 = [0,1+i*2]        
        data_end = [data_start[0] + 25,i*2]
        data_end2 = [data_start2[0] + 25,i*2+1]
        i+=1
        if measure == 'Loss':
            worksheet.write_column(*data_start, data=file.Loss)
            worksheet.write_column(*data_start2, data=file.Val_Loss)
        elif measure == 'Accuracy':
            worksheet.write_column(*data_start, data=file.Accuracy)
            worksheet.write_column(*data_start2, data=file.Val_Accuracy)
        chart.add_series({
            'values': [worksheet.name] + data_start + data_end,
            'name': architecture,})
        chart2.add_series({
            'values': [worksheet.name] + data_start2 + data_end2,
            'name': architecture,})
        
    worksheet.insert_chart('I1', chart)  
    worksheet.insert_chart('I16', chart2)        
    workbook.close() 
    
generate_configuration_mean_training_graph('Loss', 'random weights')
generate_configuration_mean_training_graph('Accuracy', 'random weights')

generate_configuration_mean_training_graph('Loss', 'tf, unfreezed')
generate_configuration_mean_training_graph('Accuracy', 'tf, unfreezed')

generate_configuration_mean_training_graph('Loss', 'tf, half freezed' )
generate_configuration_mean_training_graph('Accuracy', 'tf, half freezed')

generate_configuration_mean_training_graph('Loss', 'tf, freezed')
generate_configuration_mean_training_graph('Accuracy', 'tf, freezed')

