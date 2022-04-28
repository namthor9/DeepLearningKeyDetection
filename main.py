# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 20:14:33 2022

@author: MaxRo
"""
import os
import glob
import scipy
import matplotlib.pyplot as plt
import librosa
from librosa import display
import numpy as np
basedir = "C:\giantsteps-key-dataset"
keydir = "C:/giantsteps-key-dataset/annotations/key/"
mp3dir = "C:/giantsteps-key-dataset/audio/"

keydict = {}
labeldict = {"C major" : 0, "C minor":1, 
             "Db major" : 2, "Db minor":3,
             "D major" : 4, "D minor":5, 
             "Eb major" : 6, "Eb minor":7,
             "E major" : 8, "E minor":9, 
             "F major" : 10, "F minor":11,
             "Gb major" : 12, "Gb minor":13,
             "G major" : 14, "G minor":15,
             "Ab major" : 16, "Ab minor":17,
             "A major" : 18, "A minor":19,
             "Bb major" : 20, "Bb minor":21,
             "B major" : 22, "B minor":23,
             }
labels = []
specs = []
lenset = set()

DONOTINCLUDE = ['1224698.LOFI','1442809.LOFI','3424038.LOFI','4452003.LOFI']
for file in os.listdir(keydir):
    f = open(keydir +file, 'r')
    name = file[0:-4]
    key = f.read()
    keydict[name] = key
i = 0

for file in os.listdir(mp3dir):
    if (file[-4:] == '.wav'):
        name = file[0:-4]
        if(name not in DONOTINCLUDE):
            print(i,name)
            audio, sr = librosa.load(mp3dir+file)
            
            hop_length = 512
            n_fft = 2048
            X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            S = librosa.amplitude_to_db(abs(X))
            S = S[:129,:1025]
            if(S.shape == (80,1025)):
                specs.append(S)
                labels.append(labeldict[keydict[name]])
            else:
                print("fail")
            plt.figure(figsize=(15, 5))
            librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
            plt.colorbar(format='%+2.0f dB')
            exit()
            i+=1

ydata = np.asarray(labels)
np.save("labeldata4.npy",ydata)
xdata = np.array(specs)
np.save("xdata4.npy",xdata)
    


