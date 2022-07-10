# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:43:35 2021

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

noise=pd.read_excel(r"C:/Users/94541/Desktop/flooding.xlsx")
mu = 0
sigma = 0.15
class_name=['a','d','c','d','e','target','f']
for m in range(7):
    for i in range(noise[class_name[m]].size):
        noise[class_name[m]][i:i+1]+=random.gauss(mu,sigma)**0.05
noise.to_excel(r"C:/Users/94541/Desktop/flooding_noise.xlsx")

    
