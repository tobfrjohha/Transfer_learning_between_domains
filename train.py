import Models.Johan_Models.mini_inception as m_i
import Models.Tobias_Models.mini_inception_resnet as m_i_r
from tensorflow.keras.models import load_model
from Training_script.transferlearning_setup import set_weights_from_model, freeze_all_conv_layers, freeze_half_model
from model_evaluation import evaluate_models_to_csv
from Dataset.datasets import boats_dataset
from Dataset.datasets import cifar_10_dataset
from model_training_script import train_model, train_model_augmented
import os

################################################TRANSFER LEARNING - CIFAR10
################################################JOHAN
#experimentor = "Johan"
#experiment = "Transferlearning_mini_inception"
#for i in range(30):
#    train_model_augmented(m_i.init_model((32,32,3),10), "cifar10", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset())

################################################TOBIAS
#experimentor = "Tobias"
#experiment = "Transferlearning_mini_inception_resnet"
#for i in range(1):
#    train_model_augmented(m_i_r.init_model((32,32,3),10), "cifar10", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset())

################################################TRAINING ON BOATS DATASET
################################################JOHAN
#experimentor = "Johan"
#experiment = "boats_mini_inception"
#for i in range(30):
#    train_model(m_i.init_model((128,128,3),5), "boats", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset())

################################################TOBIAS
#experimentor = "Tobias"
#experiment = "boats_mini_inception_resnet"
#for i in range(30):
#    train_model(m_i_r.init_model((128,128,3),5), "boats", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset())

################################################TRAINING ON BOATS DATASET WITH TRANSFER-LEARNING - ALL LAYERS FROZEN
################################################JOHAN
#experimentor = "Johan"
#experiment = "boats_mini_inception_tf_freeze"
#target_model = m_i.init_model((128,128,3), 5)
#trained_model = load_model("/home/johan/Experiment/Johan_Experiments/Transferlearning_mini_inception/saved_models/V1_/V1_model_279-0.45_best_val_model.hdf5")

#model = set_weights_from_model(trained_model, target_model)
#model = freeze_all_conv_layers(model)

#for i in range(30):
#    train_model(model, "boats", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset())

###############################################TOBIAS
#experimentor = "Tobias"
#experiment = "boats_mini_inception_resnet_tf_freeze_V2"
#folder_name = "/home/johan/Experiment/Tobias_Experiments/Transferlearning_mini_inception_resnet/saved_models/evaluation/"
#target_model = m_i_r.init_model((128,128,3), 5)
#os.chdir(folder_name)
#eval_files = os.listdir(folder_name)
#print("TRANSFER LEARNING MODEL::::::-> " + eval_files[0])
#trained_model = load_model(folder_name + eval_files[0])

#model = set_weights_from_model(trained_model, target_model)
#model = freeze_all_conv_layers(model)

#for i in range(30):
#    train_model(model, "boats", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset())

################################################TRAINING ON BOATS DATASET WITH TRANSFER-LEARNING - NO FROZEN LAYERS
################################################JOHAN
#experimentor = "Johan"
#experiment = "boats_mini_inception_tf"
#target_model = m_i.init_model((128,128,3), 5)
#trained_model = load_model = laad_model("/home/johan/Experiment/Johan_Experiments/Transferlearning_mini_inception/saved_models/V1_/V1_model_279-0.45_best_val_model.hdf5")

#model = set_weights_from_model(trained_model, target_model)

#for i in range(30):
#    train_model(model, "boats", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset())

##################################################TOBIAS
#experimentor = "Tobias"
#experiment = "boats_mini_inception_resnet_tf"
#target_model = m_i_r.init_model((128,128,3), 5)
#folder_name = "/home/johan/Experiment/Tobias_Experiments/Transferlearning_mini_inception_resnet/saved_models/evaluation/"
#os.chdir(folder_name)
#eval_files = os.listdir(folder_name)
#trained_model = load_model(folder_name + eval_files[0])

#model = set_weights_from_model(trained_model, target_model)

#for i in range(30):
#    train_model(model, "boats", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset()) 

################################################TRAINING ON BOATS DATASET WITH TRANSFER-LEARNING - HALF FROZEN
################################################JOHAN
#experimentor = "Johan"
#experiment = "boats_mini_inception_tf_HF"
#target_model = m_i.init_model((128,128,3), 5)
#trained_model = load_model("/home/johan/Experiment/Johan_Experiments/Transferlearning_mini_inception/saved_models/V1_/V1_model_279-0.45_best_val_model.hdf5")

#model = set_weights_from_model(trained_model, target_model)
#model = freeze_half_model(model)

#for i in range(30):
#    train_model(model, "boats", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset())
##################################################TOBIAS
#experimentor = "Tobias"
#experiment = "boats_mini_inception_resnet_tf_HF"
#target_model = m_i_r.init_model((128,128,3), 5)
#trained_model = load_model("/home/johan/Experiment/Tobias_Experiments/Transferlearning_mini_inception_resnet/saved_models/evaluation/V1_mini_inception_resnet_195-0.43_best_val_model.hdf5")

#model = set_weights_from_model(trained_model, target_model)
#model = freeze_half_model(model)

#for i in range(30):
#    train_model(model, "boats", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset()) 

###################################################################################V2V2V2####################################################################
################################################TRAINING ON BOATS DATASET WITH TRANSFER-LEARNING - HALF FROZEN
################################################JOHAN
#experimentor = "Johan"
#experiment = "boats_mini_inception_tf_HF_new"

#for i in range(30):
#    target_model = m_i.init_model((128,128,3), 5)
#    trained_model = load_model("/home/johan/Experiment/Johan_Experiments/Transferlearning_mini_inception/saved_models/V1_/V1_model_279-0.45_best_val_model.hdf5")
    
#    model = set_weights_from_model(trained_model, target_model)
#    model = freeze_half_model(model)
#    train_model(model, "boats", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset())
################################################TOBIAS
#experimentor = "Tobias"
#experiment = "boats_mini_inception_resnet_tf_HF_new"

#for i in range(30):
#    target_model = m_i_r.init_model((128,128,3), 5)
#    trained_model = load_model("/home/johan/Experiment/Tobias_Experiments/Transferlearning_mini_inception_resnet/saved_models/evaluation/V1_mini_inception_resnet_195-0.43_best_val_model.hdf5")
    
#    model = set_weights_from_model(trained_model, target_model)
#    model = freeze_half_model(model)
    
#    train_model(model, "boats", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset()) 
################################################TRAINING ON BOATS DATASET WITH TRANSFER-LEARNING - ALL LAYERS FROZEN
################################################JOHAN
#experimentor = "Johan"
#experiment = "boats_mini_inception_tf_freeze_v2_new"
#
#for i in range(30):
#    target_model = m_i.init_model((128,128,3), 5)
#    trained_model = load_model("/home/johan/Experiment/Johan_Experiments/Transferlearning_mini_inception/saved_models/V1_/V1_model_279-0.45_best_val_model.hdf5")
    
#    model = set_weights_from_model(trained_model, target_model)
#    model = freeze_all_conv_layers(model)
#    train_model(model, "boats", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset())

###############################################TOBIAS
#experimentor = "Tobias"
#experiment = "boats_mini_inception_resnet_tf_freeze_V2_new"

#for i in range(30):
#    folder_name = "/home/johan/Experiment/Tobias_Experiments/Transferlearning_mini_inception_resnet/saved_models/evaluation/"
#    target_model = m_i_r.init_model((128,128,3), 5)
#    os.chdir(folder_name)
#    eval_files = os.listdir(folder_name)
#    print("TRANSFER LEARNING MODEL::::::-> " + eval_files[0])
#    trained_model = load_model(folder_name + eval_files[0])
    
#    model = set_weights_from_model(trained_model, target_model)
#    model = freeze_all_conv_layers(model)
#    train_model(model, "boats", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset())

################################################TRAINING ON BOATS DATASET WITH TRANSFER-LEARNING - NO FROZEN LAYERS
##################################################TOBIAS
#experimentor = "Tobias"
#experiment = "boats_mini_inception_resnet_tf_new"
#folder_name = "/home/johan/Experiment/Tobias_Experiments/Transferlearning_mini_inception_resnet/saved_models/evaluation/"
#os.chdir(folder_name)
#eval_files = os.listdir(folder_name)

#for i in range(30):
#    target_model = m_i_r.init_model((128,128,3), 5)

#    trained_model = load_model(folder_name + eval_files[0])
    
#    model = set_weights_from_model(trained_model, target_model)
#    train_model(model, "boats", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset()) 

################################################JOHAN
#experimentor = "Johan"
#experiment = "boats_mini_inception_tf_new"
#trained_model = load_model = load_model("/home/johan/Experiment/Johan_Experiments/Transferlearning_mini_inception/saved_models/V1_/V1_model_279-0.45_best_val_model.hdf5")

#for i in range(2, 30):
#    target_model = m_i.init_model((128,128,3), 5)
    
#    model = set_weights_from_model(trained_model, target_model)
#    train_model(model, "boats", i,experimentor, experiment)
#evaluate_models_to_csv(experimentor, experiment, boats_dataset())

