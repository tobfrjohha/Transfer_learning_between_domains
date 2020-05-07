# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:51:00 2020

@author: combitech
"""

import tensorflow as tf
from tensorflow import reshape
from tensorflow.keras import models
from tensorflow.keras import backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

def grad_cam(model, layer_name, img_location, hm_alpha, dim, name, network_name): 
    im = img_location 
    image = cv2.imread(im, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
    img_tensor = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = img_tensor/255.
    
    def preprocess(img):
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img,axis=0)
        return img
    
    image_1 = preprocess(img_tensor)
    
    predict = model.predict(image_1)
    target_class = np.argmax(predict[0])
    print("Target Class = %d"%target_class)
    #  print(predict[0])
    print(predict[0][np.argmax(predict[0])])
    last_conv = model.get_layer(layer_name)
    heatmap_model = models.Model([model.inputs], [last_conv.output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = heatmap_model(image_1)
        loss = predictions[:, np.argmax(predictions[0])]  
        grads = tape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads,axis=(0,1,2))     

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1) #binds different channels to one as a mean value
    heatmap = np.maximum(heatmap,0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    
    heatmap = reshape(heatmap, (heatmap.shape[1],heatmap.shape[2]), 3)
    plt.imshow(heatmap)
    
    #Restore image dimensions
    #image[:,:, 1].shape[1] -->> Corresponds to the y-axis of the img dimensions,
    #image[:,:, 1].shape[0] -->> Corresponds to the x-axis of the img dimensions
    heatmap = np.expand_dims(heatmap,axis=-1)
    upsample = cv2.resize(heatmap, (image[:,:, 1].shape[1],image[:,:, 1].shape[0]), 3)
    #upsample = cv2.cvtColor(upsample, cv2.COLOR_BGR2RGB)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(upsample)
    plt.savefig("C:/Users/combitech/Desktop/Git/CNN/Jupyter/gradcam images/" + name + "_heatmap.jpg", bbox_inches='tight', pad_inches=0)
    
    plt.xticks([])
    plt.yticks([])
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(image)    
    plt.savefig("C:/Users/combitech/Desktop/Git/CNN/Jupyter/gradcam images/" + name + "_clean.jpg", bbox_inches='tight', pad_inches=0)
    plt.imshow(upsample,alpha=hm_alpha)
    plt.savefig("C:/Users/combitech/Desktop/Git/CNN/Jupyter/gradcam images/" + name + ".jpg", bbox_inches='tight', pad_inches=0)
    plt.show()
    
def printHMvariations(model, layer, img,dim, name, network_name):
    IMG = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    alpha=1.0
    for i in range(2):
        grad_cam(model, layer, img, alpha, dim, name, network_name)
        alpha -= 0.4
    IMG = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    plt.imshow(IMG)
    plt.show()