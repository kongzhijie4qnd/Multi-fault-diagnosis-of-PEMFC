# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:03:50 2021

@author: 94541
"""
from tensorflow import keras
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
import numpy as np
import os


### dividing into groups 
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
def build_model():
  model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(6,)),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.Adam(0.01)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

df=pd.read_excel(r"C:/Users/94541/Desktop/normal2.xlsx")
df_labels=df.pop('target')
df_normalization=normalization(df)
for i in range(624):
    min_df= df_normalization[i*10:(i+1)*10]
    locals()['normal_train_dataset'+str(i)]=min_df
for i in range(624):
    min_labels= df_labels[i*10:(i+1)*10]
    locals()['normal_train_labels'+str(i)]=min_labels
model=build_model()
for m in range(2):
    print("{}次训练".format(m))
    model.load_weights(r'C:\Users\94541\Desktop\checkpoint./normal')
    a= locals()['normal_train_dataset'+str(m)]
    b= locals()['normal_train_labels'+str(m)]
    locals()['chang'+str(m)]=[]

    for n in range(10):  
        model.fit(a[n:n+1],b[n:n+1],
                  epochs=1, verbose=1)
        
        w,b=model.layers[0].get_weights()
        w_,b_=model.layers[1].get_weights()
        w=w.flatten()
        w_=w_.flatten()
        temp=np.hstack((w,b,w_,b_))
        locals()['chang'+str(m)]=np.hstack((locals()['chang'+str(m)],temp))
   
        

