# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 21:54:51 2016

@author: calm
"""
import glob, cv2
import numpy as np
from keras.models import model_from_json
from anaData import modelDict


def logoPredictor(path, rows, cols):
    model = model_from_json(open(path+'logo_architecture.json').read())
    model.load_weights(path+'logo_weights.h5')
    model.compile(optimizer = 'adam', loss='categorical_crossentropy',metrics=['accuracy'])
    
    imgs = loadImgs(path+'logo_test/', rows, cols)
    imgs = imgs.reshape(imgs.shape[0], 1, rows, cols)
    
    classes = model.predict(imgs)
    _, model_Dict = modelDict(path)
    output = []
    for cls in classes:
        output.append(model_Dict.keys()[model_Dict.values().index(cls)])
        
    return output #A Numpy array of predictions

def loadImgs(imgsfolder, rows, cols):
    myfiles = glob.glob(imgsfolder+'*.jpg', 0)
    nPics = len(myfiles)
    X = np.zeros((nPics, rows, cols), dtype = 'uint8')
    i = 0;
    for filepath in myfiles:
        temp = cv2.imread(filepath, 0)
        if temp == None:
            continue
        elif temp.size < 1000:
            continue
        elif temp.shape == [rows, cols, 1]:
            X[i,:,:] = temp
        else:
            X[i,:,:] = cv2.resize(temp,(cols, rows), interpolation = cv2.INTER_CUBIC)
        i += 1
    return X
