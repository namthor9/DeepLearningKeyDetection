# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:12:58 2022

@author: MaxRo







"""
import numpy
import scipy
import sklearn
from sklearn import model_selection,preprocessing
import numpy as np
import pandas as pd
import os
import tensorflow
from tensorflow import keras,math
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, Dense, MaxPooling2D, Conv2D, Average
from datetime import datetime


datadir= "C:/Users\MaxRo\OneDrive\Desktop\junioryear\ece485/xdata2.npy"
labeldir = "C:/Users\MaxRo\OneDrive\Desktop\junioryear\ece485/labeldata2.npy"

imwidth = 1025
imheight = 129
xdata = numpy.load(datadir)
labels = numpy.load(labeldir)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(xdata, labels, test_size=0.2)
x_train = x_train.reshape(x_train.shape[0], imheight, imwidth, 1)
x_test = x_test.reshape(x_test.shape[0], imheight, imwidth, 1)
input_shape = (imheight, imwidth, 1)


model = tensorflow.keras.models.Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(imheight,imwidth,1),strides=1))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',strides=1))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',strides=1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(24, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=4, verbose=1)
	# evaluate model
accuracy = model.evaluate(x_test, y_test, batch_size=4, verbose=1)
predictions = model.predict(x_test)
predicted_classes = numpy.argmax(predictions, axis=1)
confusion = tensorflow.math.confusion_matrix(y_test,predictions,num_classes=24)
