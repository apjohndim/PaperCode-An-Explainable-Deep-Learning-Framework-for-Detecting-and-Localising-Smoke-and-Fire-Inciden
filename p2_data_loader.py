# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:51:50 2021

@author: John
"""


''' LOAD BASIC LIBRARIRES'''

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from imutils import paths
import numpy as np
import random
import cv2
import os
from PIL import Image 
import numpy
import tensorflow as tf
import pandas as pd
from scipy.ndimage import rotate
import logging
SEED = 2   # set random seed


logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level = logging.DEBUG, filename = 'C:\\Users\\User\\ZZZ. FIRE\\main_logger.log')
logging.captureWarnings(True)
# print = logging.info

def normalize_from_pixels (scan):
    
    MIN_BOUND = scan.min()
    MAX_BOUND = scan.max()
    
    scan = (scan - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    scan[scan>1] = 1.
    scan[scan<0] = 0.
    return scan

def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img


def gaussian_noise(img, mean=0, sigma=0.03):
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 1.0
    mask_overflow_lower = img+noise < 0
    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = 0
    img += noise
    return img
#imgplot = plt.imshow(img)


def make_square(img,s):
    
    s1 = max(img.shape[0:2])
    #Creating a dark square with NUMPY  
    f = np.zeros((s1,s1,3),np.uint8)
    
    #Getting the centering position
    ax,ay = (s1 - img.shape[1])//2,(s1 - img.shape[0])//2
    
    #Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img
    f = cv2.resize(f, (s, s))
    return f

def load_fire (path, in_shape, verbose):
    
    #LOADS 3IN1 IMAGES (DUAL + SUBSTRACTION). NO PERIOXES

    width = in_shape[1]
    height = in_shape[0]
    if verbose:
        print("[INFO] loading images")
        
        
    data_early = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths_early = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths_early) # Shuffle the image data
    
    # loop over the input images
    for l,imagePath_early in enumerate(imagePaths_early[:15000]): #load, resize, normalize, etc
        if (l % 2) == 0:
            continue
        try:
            if verbose:
                print("Preparing Image: {}".format(imagePath_early))
            
            #logging.warning(imagePath_early)
            image = cv2.imread(imagePath_early)
            image = make_square(image,s=in_shape[0])
            image = normalize_from_pixels (image)
    
            # plt.imshow(image)
            # plt.show
            
            lab = os.path.basename(os.path.dirname(imagePath_early))
            
            if lab != 'NEUTRAL':
                lab = 'PoFS'
    
            data_early.append(image)
            labels.append(lab)
        except Exception as e:
            print (e)
            os.remove(imagePath_early)
            print(imagePath_early)
            continue
            
    data_early = np.array(data_early, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = tf.keras.utils.to_categorical(labels, num_classes=2)
    print("Data and labels loaded and returned")
    return data_early, labels, image






