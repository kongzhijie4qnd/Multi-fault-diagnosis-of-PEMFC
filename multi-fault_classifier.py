# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:50:19 2021

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
import kerastuner as kt

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
'''
dataset=pd.read_excel(r"C:/Users/94541/Desktop/changewithnoise1.xlsx")
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_labels = train_dataset.pop('labels')
test_labels = test_dataset.pop('labels')
train_dataset=normalization(train_dataset)
test_dataset=normalization(test_dataset)
train_datase=train_dataset.fillna(0)
test_datase=test_dataset.fillna(0)
'''
def build_classification_model():
  model = keras.Sequential([
    layers.Dense(784, activation='relu', input_shape=(810,)),
    layers.Dense(4)
  ])
  model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  return model
checkpoint_path = r"C:/Users/94541/Desktop/check/trainningforreal/cp1-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq='epoch')
model=build_classification_model()
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
history=model.fit(train_datase, train_labels, epochs=98,validation_split=0.2,callbacks=stop_early,shuffle=True)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

def model_builder(hp):
  model = keras.Sequential()
  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=1024, step=32) 
  model.add(keras.layers.Dense(units= hp_units, activation='relu',input_shape=(810,)))
  model.add(keras.layers.Dense(4))
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir21',
                     project_name='intro_to_kt21')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(train_dataset, train_labels, epochs=100, validation_split=0.2, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model.save_weights(r'C:/Users/94541/Desktop/lunwencailiao/classification')

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learnig_rate')}
""")
'''
'''
1 best  0.8847
2 best  0.8863
'''

predictions = model.predict(test_dataset)
score = tf.nn.softmax(predictions)
score=np.array(score)
prediction=[]
for i in range(1043):
    max=score[i,:].max()
    if score[i,0]==max:
        prediction.append(0)
    elif score[i,1]==max:
       prediction.append(1)
    elif score[i,2]==max:
        prediction.append(2)
    elif score[i,3]==max:
        prediction.append(3)
average_precision_score(test_labels,prediction)
results = model.evaluate(test_dataset,  test_labels, verbose=2)
recall_score(test_labels,prediction)
for i in range(1043):
    if prediction[i]==0:
        prediction_focus_0.append(1)
    else:
        prediction_focus_0.append(0)
for i in range(1043):
    if test_labels[i]==0:
        test_labels_focus_0.append(1)
    else:
        test_labels_focus_0.append(0)
    

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('accuracy')    
  plt.plot(hist['epoch'], hist['accuracy'],
           label='Train accuracy')
  plt.plot(hist['epoch'], hist['val_accuracy'],
           label = 'val accuracy')
  plt.ylim([0,10])
  plt.legend()
  plt.figure()
  plt.show()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('accuracy')    
  plt.plot(hist['epoch'], hist['accuracy'],
           label='Train accuracy')
  plt.plot(hist['epoch'], hist['val_accuracy'],
           label = 'val accuracy')
  plt.ylim([0,1])
  plt.legend()
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('loss')
  plt.plot(hist['epoch'], hist['loss'],
         label='Train Loss')
  plt.plot(hist['epoch'], hist['val_loss'],
         label = 'val Loss')
  plt.ylim([0,1]) 
  plt.legend()
  plt.show()
plot_history(history)

