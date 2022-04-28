# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:05:18 2022

@author: MaxRo
"""
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
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, Dense, MaxPooling2D, Conv2D, Average,AveragePooling2D
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

datadir= "C:/Users\MaxRo\OneDrive\Desktop\junioryear\ece485/xdata4.npy"
labeldir = "C:/Users\MaxRo\OneDrive\Desktop\junioryear\ece485/labeldata4.npy"

imwidth = 1025
imheight = 80
xdata = numpy.load(datadir)
labels = numpy.load(labeldir)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(xdata, labels, test_size=0.2)
x_train = x_train.reshape(x_train.shape[0], imheight, imwidth, 1)
x_test = x_test.reshape(x_test.shape[0], imheight, imwidth, 1)
input_shape = (imheight, imwidth, 1)



model = tensorflow.keras.models.Sequential()
model.add(Conv2D(filters=8, kernel_size=(5,5), activation='elu', input_shape=(imheight,imwidth,1),strides=1,padding='same'))
model.add(Conv2D(filters=8, kernel_size=(5,5), activation='elu',strides=1,padding='same'))
# model.add(Conv2D(filters=8, kernel_size=(5,5), activation='elu',strides=1,padding='same'))
# model.add(Conv2D(filters=8, kernel_size=(5,5), activation='elu',strides=1,padding='same'))
#model.add(TimeDistributed(Average()), kwargs)
model.add(AveragePooling2D(pool_size=(1,1025),padding='same'))
#model.add(AveragePooling2D(pool_size=(1,2000),padding='same'))
model.add(Flatten())
model.add(Dense(48, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(24, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=11, batch_size=4, verbose=1)
	# evaluate model
accuracy = model.evaluate(x_test, y_test, batch_size=4, verbose=1)
predictions = model.predict(x_test)
predicted_classes = numpy.argmax(predictions, axis=1)
# confusion = tensorflow.math.confusion_matrix(y_test,predicted_classes,num_classes=24).
# con_mat = tensorflow.math.confusion_matrix(y_test,predicted_classes,num_classes=24).numpy()

# con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
# classes = np.arange(0,24)

# con_mat_df = pd.DataFrame(con_mat_norm,
#                      index = classes, 
#                      columns = classes)

# figure = plt.figure(figsize=(8, 8))
# sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
# plt.tight_layout()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()
