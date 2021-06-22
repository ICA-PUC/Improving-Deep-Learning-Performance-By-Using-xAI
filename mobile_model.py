from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout,BatchNormalization,LeakyReLU,GlobalAveragePooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
#from keras_preprocessing.image import ImageDataGenerator

from keras import applications
from keras import backend as K
import numpy as np
from sklearn.utils import shuffle

from glob import glob
from os.path import join,basename 
import re
import os
import cv2
from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import argparse
import csv
from glob import glob
import hashlib
import sys
import pandas as pd
import tensorflow as tf

from class_callback import TestCallback

def main(args):
  '''
  Train Model for tensorflow 1.x Used at work:

  Improving Deep Learning Performance By Using Explainable Artificial Intelligence (xAI) Approaches

  Network used Mobile 

  
  
  
  
  '''



  def resnet_or_mobile():
    # If you want to use your our weight pre trained##
    #weights_res='/share_alpha/Submarina/pre_treined_features/new_dataset/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    #resnet_50_model = applications.ResNet50(weights='imagenet', include_top=True)#,input_tensor=input_tensor)
    
    mobile = applications.MobileNet(weights='imagenet', include_top=True)
    mobile.summary()
    return mobile
  
  if args.pre_treinada=='mobile':
    net = resnet_or_mobile()

  
  def create_custom_model(net,optm):
    

    # Change here output of resnet"
    x=net.layers[-2].output
    
    #x = Dense(256, activation='relu')(x)
    pred = Dense(10, activation='softmax')(x)
    
    custom_model = Model(net.input,pred)
    
    #print('frozen layers custom network')
    
    for layer in custom_model.layers:
      layer.trainable = True
    #print('Free the last 2 layers custom network ')
    #for layer in custom_model.layers[-1:]:
    #  print(layer.name)
    #  layer.trainable = True

    if args.weight=='sim':
      # Change the weights here[256,3.1,12,2,12]
      weights = [1,2,0.2,1.5,0.2,1.5,1,2,2,2]
      loss = custom_loss(weights)
      custom_model.compile(loss = loss, optimizer=optm , metrics=['accuracy'])

    if args.weight=='nao':
      custom_model.compile(loss = 'categorical_crossentropy', optimizer=optm , metrics=['accuracy',f1_score])
    

    return custom_model

  def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))      
  def custom_loss(weights):    
    weights = K.variable(weights)
    
    def loss(y_true, y_pred):
      # scale predictions so that the class probas of each sample sum to 1
      y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
      # clip to prevent NaN's and Inf's
      y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
      loss = y_true * K.log(y_pred) + (1-y_true) * K.log(1-y_pred)
      loss = loss * weights 
      loss = - K.mean(loss, -1)
      return loss
    return loss

  def Train(net, train,validation ,batch_size, epochs,steps_per_epoch_train,steps_per_epoch_valid,test_generator,steps_per_epoch_test): 

    print('Start training')
    # define file name to save the model
    
    file_name =args.results_path+'/mobile_new_'+args.index +'.h5'
    #file_name ='placa_results_new/generative/mobile_generative_new.h5'
    # create checkpoint callbacks to save best model for the validation set
    checkpointer = ModelCheckpoint(file_name, monitor='val_f1_score', verbose=1, save_best_only=True,mode='max')
    # create early stopp callbacks to stop training after not imporvment
    early_stop = EarlyStopping(monitor = 'val_f1_score', min_delta = 0.001, 
                                mode = 'max', patience =60)     
    #patience 60
    csv_logger = CSVLogger(args.results_path+'/mobile_train_'+args.index +'.csv', append=True)

    #callbacks=[early_stop,checkpointer,csv_logger,TestCallback(test_generator,steps_per_epoch_test)]
    custom_callback=TestCallback(test_generator,steps_per_epoch_test,args.results_path+'/mobile_test_'+args.index+'.csv')
    print("-----Not Using Data augmentation---------------") 
    net.fit_generator(generator=train,
                    use_multiprocessing=True,
                    epochs=epochs,
                    verbose=1,
                    steps_per_epoch=steps_per_epoch_train,
                    validation_steps=steps_per_epoch_valid,
                    validation_data =validation,
                    callbacks=[early_stop,checkpointer,csv_logger,custom_callback])
    
  
####################################################################################
#PARAMETROS TREINO
##################################################################
  # Change here parameters
  lr = 1e-3
  batch_size = 16
  epochs = 450
  opt = Adam(lr = lr)
  #patience epochs 200
  
  custom_model = create_custom_model(net,opt)
  custom_model.summary()
  
  
  datagen=ImageDataGenerator(validation_split=0.25,preprocessing_function=applications.mobilenet.preprocess_input)
  datagen_test=ImageDataGenerator(preprocessing_function=applications.mobilenet.preprocess_input)  

  train_generator = datagen.flow_from_directory(
    directory=args.dataset_folder_train+"/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset='training'
    )
  
  valid_generator = datagen.flow_from_directory(
    directory=args.dataset_folder_train+"/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset='validation'
    )
  test_generator = datagen_test.flow_from_directory(
    directory=args.dataset_folder_test+"/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    )
 
  steps_per_epoch_train = len(train_generator)

  steps_per_epoch_valid = len(valid_generator)

  steps_per_epoch_test = len(test_generator)

  print('passos',steps_per_epoch_test)
  
  

  Train(custom_model, train_generator,valid_generator, batch_size, epochs,steps_per_epoch_train,steps_per_epoch_valid,test_generator,steps_per_epoch_test)
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=' train explain AI')
  parser.add_argument('--pre_treinada', type=str, dest='pre_treinada', help='mobile')
  parser.add_argument('--pesos', type=str, dest='weight', help='Weight yes or no{sim ou nao} nao for same result of paper')
  parser.add_argument('--dataset_folder_train', type=str, dest='dataset_folder_train', help=' Train Diretory with subfolder of class')
  parser.add_argument('--dataset_folder_test', type=str, dest='dataset_folder_test', help=' Test Diretory with subfolder of class')
  parser.add_argument('--results', type=str, dest='results_path', help='Folder to savel the model and logs output')
  parser.add_argument('--index', type=str, dest='index', help='index to the model and csv. Should be any int number')
  
  args = parser.parse_args()

  main(args)
#python mobile_model.py --pre_treinada mobile --pesos nao --dataset_folder_train images_train --dataset_folder_test images_test --results placa_results_new/generative --index
 