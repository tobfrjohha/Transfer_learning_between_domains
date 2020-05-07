
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from path import Path
import random
import math

import os
import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from PIL import Image

from shutil import copy
    
def create_spectrogram(filename, name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename ='./Data/Train/' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, name, clip, sample_rate, fig, ax, S
    
   
def create_spectrogram_test(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = Path('./Data/Test/' + name + '.jpg')
    fig.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S
  
def create_spectrogram_sample(source_file, destination, name, i, frame, window, number_of_samples, n_mels):
    plt.interactive(False)
    clip, sample_rate = librosa.load(source_file, sr=None)
    for sample in range(number_of_samples):                 
        samp = clip[i+sample*frame:i+sample*frame+frame]
        if(len(samp)== frame):
            print(len(samp))
            fig = plt.figure(figsize=[0.72, 0.72])
            ax =fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            S = librosa.feature.melspectrogram(y=samp, sr=sample_rate, n_mels=n_mels,
                                              hop_length=window)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
            filename = Path(destination + '/' + name + str(sample) + '.jpg')
            fig.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close()
            fig.clf()
            plt.close(fig)
            plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S  
    

def create_spectrograms_from_folder(source_folder, destination_folder, name, frame_size, n_mels, hop_length, win_length, n_fft):
    plt.interactive(False)
    index = 0
    for audio_file in os.listdir(source_folder):
        name = name = os.path.splitext(audio_file)[0]
        audio_file = os.path.join(source_folder, audio_file)
        data, sr = librosa.load(audio_file)
        for sample in range(math.floor(len(data)/frame_size)):
            y = data[frame_size*sample:frame_size*sample + frame_size]                 
            if(len(y)== frame_size):
                print(math.floor(len(data)/frame_size))
                fig = plt.figure(figsize=[0.72, 0.72])
                ax =fig.add_subplot(111)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_frame_on(False)
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                       fmax=12000, hop_length=hop_length,
                                       win_length=win_length, n_fft=n_fft)
                
                librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
                filename = Path(destination_folder + '/' + name + str(sample)+ str(index) + '.jpg')
                fig.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
                plt.close()
                fig.clf()
                plt.close(fig)
                plt.close('all')
                index = index + 1
    #del filename,name,data,sample_rate,fig,ax,S  

def create_spectrograms_from_folder_with_array(source_folder, destination_folder, name, frame_size, n_mels, hop_length, win_length, n_fft):
    plt.interactive(False)
    index = 0
    arr = np.array()
    for audio_file in os.listdir(source_folder):
        name = name = os.path.splitext(audio_file)[0]
        audio_file = os.path.join(source_folder, audio_file)
        data, sr = librosa.load(audio_file)
        for sample in range(math.floor(len(data)/frame_size)):
            y = data[frame_size*sample:frame_size*sample + frame_size]                 
            if(len(y)== frame_size):
                print(math.floor(len(data)/frame_size))
                fig = plt.figure(figsize=[0.72, 0.72])
                ax =fig.add_subplot(111)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_frame_on(False)
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                       fmax=12000, hop_length=hop_length,
                                       win_length=win_length, n_fft=n_fft)
                
                librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
                filename = Path(destination_folder + '/' + name + str(sample)+ str(index) + '.jpg')
                fig.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
                plt.close()
                fig.clf()
                plt.close(fig)
                plt.close('all')
                index = index + 1

def generate_dataset_from_folders(path, batch_size, img_height, img_width):
   
    classes = os.listdir(path)  
    #x_train =[path + i for i in os.listdir(path)]
    
    data_dir = pathlib.Path(path)
    class_names = np.array([item.name for item in data_dir.glob('*')])
    image_count = len(list(data_dir.glob('*/*.jpg')))
    
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    #STEPS_PER_EPOCH = np.ceil(image_count/batch_size)
    
    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         target_size=(img_height, img_width),
                                                         classes = list(class_names))
                                                        # classes = list(class_names), class_mode="sparse")
    return train_data_gen



def get_images_from_folder(folder_path, width, height):
    images = []
    for img in os.listdir(folder_path):
        img = os.path.join(folder_path, img)
        img = image.load_img(img, target_size=(width, height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)
    images = np.vstack(images)

    return images

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(len(label_batch)):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(label_batch[n])
      plt.axis('off')
      
      
def randomly_split_data(root_path, new_path1, new_path2, ratio):
    for classes in list(pathlib.Path(root_path).glob('*')):
        print(classes)
        all_files = list(pathlib.Path(classes).glob('*.jpg'))
        random.shuffle(all_files)
        for file in all_files[0:math.ceil(len(all_files)/ratio)]:
            copy(file, new_path1 + os.path.basename(classes) + "\\")
        for file in all_files[math.ceil(len(all_files)/ratio):len(all_files)]:
            copy(file, new_path2 + os.path.basename(classes) + "\\")


