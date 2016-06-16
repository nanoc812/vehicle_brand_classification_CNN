# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 01:20:48 2016

@author: calm
"""


import numpy as np
import cv2, glob, string
import os, random


def loadLogoSet(path, rows,cols,test_data_rate=0.15):
    random.seed(612)
    _, imgID = readItems('data.txt')
    y, _ = modelDict(path)
    nPics =  len(y)
    faceassset = np.zeros((nPics,rows,cols), dtype = np.uint8) ### gray images
    noImg = []
    for i in range(nPics):
        temp = cv2.imread(path +'logo/'+imgID[i]+'.jpg', 0)
        if temp == None:
            noImg.append(i)
        elif temp.size < 1000:
            noImg.append(i)
        else:
            temp = cv2.resize(temp,(cols, rows), interpolation = cv2.INTER_CUBIC)
            faceassset[i,:,:] = temp
    y = np.delete(y, noImg,0); faceassset = np.delete(faceassset, noImg, 0)
    nPics = len(y)
    index = random.sample(np.arange(nPics), int(nPics*test_data_rate))
    x_test = faceassset[index,:,:]; x_train = np.delete(faceassset, index, 0)
    y_test = y[index]; y_train = np.delete(y, index, 0)
    return (x_train, y_train), (x_test, y_test)


def imgSeg(img):
    approx = imgSeg_contour(img, 4,4,4, 0.04)
    himg, wimg , _ = img.shape[:3]
    #h1, h2, w1, w2 = imgSeg_rect(approx, himg, wimg)
    h1, h2, w1, w2 = imgSeg_logo(approx, himg, wimg)
    if (w2-w1) < 20:
        approx = imgSeg_contour(img, 6, 6, 6, 0.02)
        himg, wimg , _ = img.shape[:3]
        #h1, h2, w1, w2 = imgSeg_rect(approx, himg, wimg)
        h1, h2, w1, w2 = imgSeg_logo(approx, himg, wimg)
    if (h2-h1) > (w2-w1): 
        approx = imgSeg_contour(img, 2,2,2, 0.04)
        himg, wimg , _ = img.shape[:3]
        #h1, h2, w1, w2 = imgSeg_rect(approx, himg, wimg)
        h1, h2, w1, w2 = imgSeg_logo(approx, himg, wimg)
    #cv2.rectangle(img,(w1, h1), (w2,h2), 255, 2)
    return img[h1:h2, w1:w2,:]


def imgSeg_logo(approx, himg, wimg):
    w = np.amax(approx[:,:,0])-np.amin(approx[:,:,0]); h = np.amax(approx[:,:,1])-np.amin(approx[:,:,1])
    if float(w)/float(h+0.001) > 4.5:
        h = int(float(w)/3.5)
    w0 = np.amin(approx[:,:,0]); h0 = np.amin(approx[:,:,1])
    h1 = h0-int(3.5*h); h2 = h0;
    w1 = max(w0+w/2-int(0.5*(h2-h1)), 0); w2 = min(w0+w/2+int(0.5*(h2-h1)), wimg-1)
    return h1, h2, w1, w2


def imgSeg_rect(approx, himg, wimg):
    w = np.amax(approx[:,:,0])-np.amin(approx[:,:,0]); h = np.amax(approx[:,:,1])-np.amin(approx[:,:,1])
    if float(w)/float(h+0.001) > 4.5:
        h = int(float(w)/3.5)
    w0 = np.amin(approx[:,:,0]); h0 = np.amin(approx[:,:,1])
    h1 = h0-int(3.6*h); h2 = min(h0+int(3*h), himg-1)
    w1 = max(w0+w/2-(h2-h1), 0); w2 = min(w0+w/2+(h2-h1), wimg-1)
    return h1, h2, w1, w2

def imgSeg_contour(img, b,g,r, per):
    lower = np.array([0, 0, 0])
    upper = np.array([b,g,r])
    shapeMask = cv2.inRange(img, lower, upper)

    #http://stackoverflow.com/questions/27746089/python-computer-vision-contours-too-many-values-to-unpack
    _, cnts, hierarchy = cv2.findContours(shapeMask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:4]
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, per * peri, True) ### 0.04 ###
        if (len(approx) >= 4) and (len(approx) < 6):
            break
    return approx

def modelDict(path):
    vmc, imgID = readItems('data.txt')
    nPics =  len(vmc[:,1])
    model_dict = {}
    count = int(0);
    for i in range(nPics):
        if vmc[i,1] not in model_dict:
            #print count
            model_dict[int(vmc[i,1])] = count
            count = count + 1
    
    model_Y = np.zeros((nPics,1), dtype = np.uint8)
    for i in range(nPics): 
        model_Y[i] = model_dict[vmc[i,1]]
    return model_Y, model_dict


def readItems(data_dir):
    fr = open(data_dir,'r')
    alllines = fr.readlines()
    num = len(alllines)
    cmv = np.zeros((num,3))
    imgID = []
    for i in range(num):
        line = alllines[i]
        temp = string.replace(line,'\r','');temp = string.replace(temp,'\n',''); temp = temp.split(' ')
        imgID.append(temp[2])
        cmv[i,:] = [temp[0],temp[1],temp[3]]
    return cmv, imgID

def imgPreprocess(img_dir):
    myfiles = glob.glob(img_dir+'*.jpg')
    
    temp = img_dir.split('/')
    newDir = '/'.join(temp[:(len(temp)-2)])
    if not os.path.exists(newDir+'/logo/'):
        os.mkdir(newDir+'/logo/')
    
    for filepath in myfiles:
        img = cv2.imread(filepath)
        logo = imgSeg(img)
        sd = filepath.rfind('/'); ed = filepath.find('.'); filename = filepath[int(sd+1):int(ed)]
        cv2.imwrite(newDir+'/logo/'+filename+'.jpg',logo)
        print("car logo segmentation success,%s"%filename)

def imgResize(img, n_rows, n_cols, flag = 1):
    h,w,_ = img.shape
    if flag == 0: ### turn the colorful imput into a gray image
        img = cv2.cvtColor(img, cv2.CV_BGR2GRAY)
    #print h, w
    if w < h:
        #newIMG = img[(h-w):,:]
        newIMG = cv2.resize(img[(h-w):,:],(n_rows, n_cols), interpolation = cv2.INTER_AREA)
    else:
        d = int(round((w-h)/2))
        #newIMG = img[:,d:(w-d)]
        newIMG = cv2.resize(img[:,d:(w-d)],(n_rows, n_cols), interpolation = cv2.INTER_AREA)
    return newIMG


def imgThresh(img):
    newIMG = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv2.THRESH_BINARY,7,2)
    return newIMG
    
if __name__ == "__main__":
    
    imgPreprocess('/home/calm/Documents/CARS/train/subset/test/')
