from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout,BatchNormalization,LeakyReLU,GlobalAveragePooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator


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
from sklearn.metrics import f1_score as f1_S
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
#from efficientnet import EfficientNetB0 as Net

def main(args):

    '''
  Test code for tensorflow 1.x Used at work:

  Improving Deep Learning Performance By Using Explainable Artificial Intelligence (xAI) Approaches

   Mobile 

  
  
  
  
  '''
 
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
  
  
####################################################################################
#PARAMETROS TREINO
##################################################################
  # Change here parameters
    lr = 1e-3
    batch_size = 1
    epochs = 450
    opt = Adam(lr = lr)
  
  


    #datagen=ImageDataGenerator(validation_split=0.25) 
    #datagen=ImageDataGenerator(validation_split=0.25,preprocessing_function=applications.resnet50.preprocess_input)
    #datagen=ImageDataGenerator(validation_split=0.25,preprocessing_function=applications.mobilenet.preprocess_input) 
    datagen_test=ImageDataGenerator(preprocessing_function=applications.mobilenet.preprocess_input)  

    test_generator = datagen_test.flow_from_directory(
    directory=args.dataset_folder_test,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    )

    steps_per_epoch_test = len(test_generator)

    

    
    print('LOAD MODEL')
    #discriminator=load_model('results_placas/generative/mobile_generative.h5', custom_objects={'LeakyReLU': LeakyReLU,'f1_score':f1_score}, compile=True)

    discriminator=load_model(args.model_name, custom_objects={'LeakyReLU': LeakyReLU,'f1_score':f1_score}, compile=True)
    dc=0
    valid_labels=[]
    predict_valid__all_img=[]
    #valid_num=int(len(valid_generator)/batch_size)
    for dc in range(steps_per_epoch_test):
        sys.stdout.write('\rDiscriminador record: ' + str(dc) + ' of ' + str(steps_per_epoch_test))
        sys.stdout.flush()

        valid_images_labels_pro=test_generator.__next__() 
        
        valid_img=valid_images_labels_pro[0]
        #print(type(valid_img))
        valid_labels+=[ valid_images_labels_pro[1]  ]

        predict_val = discriminator.predict(valid_img)

        predict_val = np.rint(predict_val)

        predict_valid__all_img+=[predict_val]
                    
    valid_labels=np.concatenate(valid_labels)
    predict_valid__all_img=np.concatenate(predict_valid__all_img)
    print('Shape label : \n')
                    #print(predict_val.shape)
                    #print(predict_valid__all_img)

                    #valid_labels=np.array(valid_labels)
                    #predict_valid__all_img=np.array(predict_valid__all_img)

    print(' \n Calculando Metricas')
                    #valid_labels=np.squeeze(valid_labels, axis=0)
                    #predict_valid__all_img=np.squeeze(predict_valid__all_img, axis=0)
    print(valid_labels.shape)
    print(predict_valid__all_img.shape)
    print(valid_labels[0])
    print(predict_valid__all_img[0])
    f1=f1_S(valid_labels, predict_valid__all_img,average='macro')
    recal=recall_score(valid_labels, predict_valid__all_img,average='macro')
    precision=precision_score(valid_labels, predict_valid__all_img,average='macro')            
    print(' f1 = {}  , recal = {} , precision = {}'.format( f1,recal,precision))
    print('\n')

    matrix_conf=confusion_matrix(valid_labels.argmax(axis=1),predict_valid__all_img.argmax(axis=1))
    print(matrix_conf)
                ### Liberando a memoria
    # Write-Overwrites
    file1 = open(args.results_folder,"w")#write mode
    file1.write(' f1 = {}  , recal = {} , precision = {}'.format( f1,recal,precision))
    file1.write('\n')
    file1.write(np.array2string(matrix_conf))
    file1.close()
                    
    valid_labels=[]
    predict_valid__all_img=[]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test explain AI')
    

    parser.add_argument('--model', type=str, dest='model_name', help='Path of  model')
    parser.add_argument('--dataset', type=str, dest='dataset_folder_test', help='Folder of dataset test')
    parser.add_argument('--results', type=str, dest='results_folder', help='Results path')
    args = parser.parse_args()
    
    main(args)

 

    #python pred.py --model 'placa_results_new/generative/mobile_new.h5' --dataset "images_test/" --results "placa_results_new/generative/all_results.txt" 