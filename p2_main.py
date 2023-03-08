# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:50:16 2021

@author: John
"""

'''CODES FOR PARATHYROID'''

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.config.list_physical_devices('GPU')


sys.path.insert(1, 'C:\\Users\\User\\ZZZ. FIRE\\')

from p2_data_loader import load_fire
from p2_main_functions import train_multi,model_save_load



#%% PARAMETER ASSIGNEMENT


'''MULTI NO FACE'''
path = 'D:\\Fire Detection Datasets\\AIO(ALL IN ONE)\\'



'''MULTI'''
in_shape  = (450,450,3) # MULTI SIAM
tune = 1 # SET: 1 FOR TRAINING SCRATCH, 0 FOR OFF THE SHELF, INTEGER FOR TRAINABLE LAYERS (FROM TUNE AND DOWN, THE LAYERS WILL BE TRAINABLE)
epochs = 10 #500 
batch_size = 4
n_split = 5 #set to 10
augmentation=True
verbose=True
class_names = ["PoFS", "Normal"]
model = 'lvgg'
classes = 2
#%%
##%% IMAGE LOAD

data, labels, image = load_fire (path, in_shape, verbose=False)


imgs_index = [0,10,101,201]

for number in imgs_index:
    image = data[number,:,:,:]
    plt.imshow(image)
    plt.show()


#%% FIT THE MODEL TO THE DATA (FOR PHASE 2)

from p2_main_functions import model_save_load,train_multi
import time
''' TRAIN - EVALUATE MODEL - GET METRICS '''


# double input
#in_shape_model = (in_shape2[0],in_shape2[1],in_shape2[2])



'''3 VGGS'''
from sklearn import preprocessing
in_shape_for_model = in_shape


# 10-FOLD
start = time.time()
#model3, group_results, fold_scores, pd_metrics,predictions_all,predictions_all_num,test_labels,labels,conf_final,cnf_matrix,conf_img,history = train_multi(data,labels=labels,epochs=epochs,batch_size=batch_size, model=model, in_shape=in_shape_for_model, tune=tune, classes=classes,n_split=n_split,augmentation=augmentation,verbose=verbose,logs=True,plot_results=False,class_names=class_names,save_variables=True)


# SAVE ONLY
model3, group_results, fold_scores, pd_metrics,predictions_all,predictions_all_num,test_labels,labels,conf_final,cnf_matrix,conf_img,history = train_multi(data,labels=labels,epochs=epochs,batch_size=batch_size, model=model, in_shape=in_shape_for_model, tune=tune, classes=classes,n_split=n_split,augmentation=augmentation,verbose=verbose,logs=True,plot_results=True,class_names=class_names,save_variables=True,save_model_just=True)


stop = time.time()
print ('Elapsed: {} seconds'.format(stop-start))

#%%

with tf.device('/CPU:0'):
    predictions_all_cat = model3.predict(data)
    predictions_all_num = model3.predict(data) #for def models functional api
    predictions_all = np.argmax(predictions_all_cat, axis=-1)
    
    
    import pandas as pd
    pd.DataFrame(predictions_all).to_csv("predictions_fire.csv")
    pd.DataFrame(labels).to_csv("labels_fire.csv")
    
    import p2_metrics
    
    THE_METRICS = p2_metrics.metrics (predictions_all, predictions_all_num, predictions_all_cat, labels, verbose = True)


model3.save('fire_model')
new_model = tf.keras.models.load_model('fire_model')


#%%

'''
LIME
'''

import p2_lime_func

# LIME COMMANDS
#items_no = [17,18, 100, 500, 600, 601, 602, 603, 604, 152]
items_no = [i for i in range (100,len(data[100:1000]),1)]
base_path = 'C:\\Users\\User\ZZZ. FIRE\\RESULTS\\LIME\\'

p2_lime_func.the_lime (items_no,predictions_all,labels,data,1,model3,verbose = False,show=False, save = True, base_path=base_path)



#%%


'''

GRAD-CAM PLUS PLUS


'''

import p2_gradcamplusplus

items_no = [i for i in range (len(data[3000:4000]))]
#items_no = [17,18, 100, 500, 600, 601, 602, 603, 604, 152]
base_path = 'C:\\Users\\User\\ZZZ. FIRE\\RESULTS\\GRADCAMPLUSPLUS\\'
p2_gradcamplusplus.gradcamplusplus (items_no,predictions_all,labels,data,1,model3,verbose = False,show=False, save = True, base_path=base_path)



