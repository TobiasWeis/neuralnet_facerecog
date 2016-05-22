#!/usr/bin/python
'''
get all files, classify them, and put them to preliminary folders to be checked by a human again
'''
import os

import cv2
import numpy as np

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import glob

from settings import *


def get_and_resize_imgs(rgb=True):
    if rgb:
        X_test = np.empty((0,3, s.img_size, s.img_size),np.float32)# contains data
    else:
        X_test = np.empty((0,1, s.img_size, s.img_size),np.float32)# contains data

    files = glob.glob("./faces_120/face_*.png") # that's were new, unclassified files are put

    for f in files:
        if rgb:
            img = cv2.resize(cv2.imread(f) / 255., (s.img_size, s.img_size))
            img = img.transpose(2,0,1).reshape(3, s.img_size, s.img_size)
            X_test = np.append(X_test, np.array([img.astype(np.float32)]), axis=0)
        else:
            img = cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY) / 255., (s.img_size, s.img_size))
            X_test = np.append(X_test, np.array([[img.astype(np.float32)]]), axis=0)

    return files, X_test

s = Settings()
net = s.net
net.load_weights_from(s.net_name)

fnames, X_test = get_and_resize_imgs(rgb=False)
preds = net.predict(X_test)

print "Predictions:"
print preds

# check if preliminary folders already exist, else create them
directories = []
for l in s.labels:
    directory = "./faces_120/prelim_%s" % (l)
    directories.append(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

# move the files to preliminary folders according to the prediction
# predictions are the indices of the labels
for i,f in enumerate(fnames):
    print "Moving %s to %s/" % (f, directories[preds[i]])
    os.system("mv %s %s" % (f, directories[preds[i]]))
