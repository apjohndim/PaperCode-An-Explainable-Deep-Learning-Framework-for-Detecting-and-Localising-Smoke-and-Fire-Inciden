# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:52:08 2021

@author: John
"""


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
#in_shape = (300, 300, 3)
#classes = 2


def make_vgg (in_shape, tune, classes): #tune = 0 is off the self, tune = 1 is scratch, tune 
    
#import pydot
    
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    

    
    for layer in base_model.layers:
        layer.trainable = False
        

    #base_model.summary()
  
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(1200, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

def make_lvgg (in_shape, tune, classes):
    
#import pydot
    
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[19:]:
        layer.trainable = True
    #base_model.summary()
    
    # early2 = layer_dict['block2_pool'].output 
    # #early2 = Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early2)
    # early2 = tf.keras.layers.BatchNormalization()(early2)
    # early2 = tf.keras.layers.Dropout(0.5)(early2)
    # early2= tf.keras.layers.GlobalAveragePooling2D()(early2)
        
    early3 = layer_dict['block3_conv4'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3 = tf.keras.layers.BatchNormalization()(early3)
    #early3 = tf.keras.layers.Dropout(0.2)(early3)
    early3= tf.keras.layers.GlobalAveragePooling2D()(early3)    
        
    early4 = layer_dict['block4_conv4'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4 = tf.keras.layers.BatchNormalization()(early4)
    #early4 = tf.keras.layers.Dropout(0.2)(early4)
    early4= tf.keras.layers.GlobalAveragePooling2D()(early4)     
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1)
    x = tf.keras.layers.concatenate([x1, early4, early3], axis=-1)  
    
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(1400, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    #model.summary()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

def make_xception (in_shape, tune, classes):
    
    base_model = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    

    for layer in base_model.layers:
        layer.trainable = False
        
    x1 = layer_dict['block14_sepconv2'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.Dense(1200, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model 

def make_inception (in_shape, tune, classes):
    
    base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    #for layer in base_model.layers:
        #print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])


    for layer in base_model.layers:
        layer.trainable = False
    


    
    x1 = base_model.output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.Dense(1500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    #print("[INFO] Model Compiled!")
    return model 

def make_resnet (in_shape, tune, classes):
    
    base_model = tf.keras.applications.ResNet152V2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    

    for layer in base_model.layers:
        layer.trainable = False
        

    
    x1 = layer_dict['post_relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(1200, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model 


def make_mobile (in_shape, tune, classes):
    
    base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    

    

    for layer in base_model.layers:
        layer.trainable = False

    
    x1 = layer_dict['out_relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.Dense(1200, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("[INFO] Model Compiled!")
    return model 


def make_dense (in_shape, tune, classes):
    
    base_model = tf.keras.applications.DenseNet201(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])

    
    for layer in base_model.layers:
        layer.trainable = False
        
    
    x1 = layer_dict['relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.Dense(1200, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("[INFO] Model Compiled!")
    return model 

def make_eff (in_shape, tune, classes):
    
    base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    
    for layer in base_model.layers:
        layer.trainable = False
    
    x1 = layer_dict['relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(1200, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model
