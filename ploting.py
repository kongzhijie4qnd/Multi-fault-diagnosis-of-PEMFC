# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:15:33 2021

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
import random

raw_normal=pd.read_excel(r"C:/Users/94541/Desktop/lunwencailiao/Normal.xlsx")

def plot_history(raw_normal):
  plt.figure()
  plt.xlabel('')
  plt.ylabel('电流/A')    
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'val Error')
  plt.ylim([0,100])
  plt.legend()
  plt.show()

plot_history(history)


plt.figure()
plt.xlabel('N')
plt.plot(raw_normal['i'],raw_normal['f'])
plt.legend()
plt.figure()
plt.show()


a=range(1,amended_normal['a'].size+1)
b=range(1,amended_starvation['a'].size+1)
c=range(1,amended_drying['a'].size+1)
d=range(1,amended_flooding['a'].size+1)
plt.figure()
plt.xlabel('N')
plt.ylabel('air compressor speed/RPM')
plt.plot(a,amended_normal['a'],label='normal')
plt.plot(b,amended_starvation['a'],label='starvation')
plt.plot(c,amended_drying['a'],label='drying')
plt.plot(d,amended_flooding['a'],label='flooding')
plt.ylim([950,1750])
plt.legend()
plt.show()

plt.figure()
plt.xlabel('N')
plt.ylabel('air compressor speed/RPM')
plt.plot(a,amended_normal['a'],label='normal')
plt.plot(b,amended_starvation['a'],label='starvation')
plt.plot(c,amended_drying['a'],label='drying')
plt.plot(d,amended_flooding['a'],label='flooding')
plt.ylim([950,1900])
plt.axes().set_aspect(10.0)
plt.legend()
plt.show()

plt.figure(figsize=(15,3))
plt.xlabel('N')
plt.ylabel('air compressor speed/RPM')
plt.plot(a,amended_normal['a'],label='normal')
plt.plot(b,amended_starvation['a'],label='starvation')
plt.plot(c,amended_drying['a'],label='drying')
plt.plot(d,amended_flooding['a'],label='flooding')
plt.ylim([950,1500])
plt.xlim([0,17000])
plt.legend()
plt.show()  


plt.figure(figsize=(15,3))
plt.xlabel('N')
plt.ylabel('air pressure/Pa')
plt.plot(a,amended_normal['b'],label='normal')
plt.plot(b,amended_starvation['b'],label='starvation')
plt.plot(c,amended_drying['b'],label='drying')
plt.plot(d,amended_flooding['b'],label='flooding')
plt.xlim([0,17000])
plt.legend()
plt.show()  


plt.figure(figsize=(15,3))
plt.xlabel('N')
plt.ylabel('hydrogen pressure/Pa')
plt.plot(a,amended_normal['c'],label='normal')
plt.plot(b,amended_starvation['c'],label='starvation')
plt.plot(c,amended_drying['c'],label='drying')
plt.plot(d,amended_flooding['c'],label='flooding')
plt.xlim([0,17000])
plt.legend()
plt.show()  


plt.figure(figsize=(15,3))
plt.xlabel('N')
plt.ylabel('opening rate of BPV')
plt.plot(a,amended_normal['d'],label='normal')
plt.plot(b,amended_starvation['d'],label='starvation')
plt.plot(c,amended_drying['d'],label='drying')
plt.plot(d,amended_flooding['d'],label='flooding')
plt.ylim([0.2,0.6])
plt.xlim([0,17000])
plt.legend()
plt.show()  

plt.figure(figsize=(15,3))
plt.xlabel('N')
plt.ylabel('exceed air coefficient')
plt.plot(a,amended_normal['e'],label='normal')
plt.plot(b,amended_starvation['e'],label='starvation')
plt.plot(c,amended_drying['e'],label='drying')
plt.plot(d,amended_flooding['e'],label='flooding')
plt.xlim([0,17000])
plt.legend()
plt.show()  

plt.figure(figsize=(15,3))
plt.xlabel('N')
plt.ylabel('voltage/V')
plt.plot(a,amended_normal['target'],label='normal')
plt.plot(b,amended_starvation['target'],label='starvation')
plt.plot(c,amended_drying['target'],label='drying')
plt.plot(d,amended_flooding['target'],label='flooding')
plt.xlim([0,17000])
plt.legend()
plt.show() 

plt.figure(figsize=(15,3))
plt.xlabel('N')
plt.ylabel('current/A')
plt.plot(a,amended_normal['f'],label='normal')
plt.plot(b,amended_starvation['f'],label='starvation')
plt.plot(c,amended_drying['f'],label='drying')
plt.plot(d,amended_flooding['f'],label='flooding')
plt.xlim([0,17000])
plt.legend()
plt.show() 

amended_normal=pd.read_excel(r"C:/Users/94541/Desktop/lunwencailiao/normalfull.xlsx")
amended_starvation=pd.read_excel(r"C:/Users/94541/Desktop/lunwencailiao/starvation.xlsx")
amended_drying=pd.read_excel(r"C:/Users/94541/Desktop/lunwencailiao/drying.xlsx")
amended_flooding=pd.read_excel(r"C:/Users/94541/Desktop/lunwencailiao/flooding.xlsx")
noise_drying=pd.read_excel(r"C:/Users/94541/Desktop/lunwencailiao/drying_noise.xlsx")