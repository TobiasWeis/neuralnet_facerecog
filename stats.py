#!/usr/bin/python
'''
display stats regarding the person and the times we saw him/her
'''

from settings import *
import glob
import matplotlib.pyplot as plt
import numpy as np

s = Settings()

folder = "./faces_120/"

hours = {}
files = {}
for label in s.labels:
    files[label] = glob.glob(folder + label + "/*.png")
    print "Got %d files for label %s" % (len(files[label]), label)
    for f in files[label]:
        # extract hour
        if f.split("-")[1][:6] != "010101":
            hour = int(f.split("-")[1][:2])
            if label in hours:
                hours[label].append(hour)
            else:
                hours[label] = []
                hours[label].append(hour)

# number of picks per label per hour
fig = plt.figure()
for i,label in enumerate(s.labels):
    try:
        fig.add_subplot(len(s.labels), 1, i+1)
        plt.hist(np.array(hours[label]), bins=range(0,24 + 1, 1))
        plt.xticks(range(24))
        plt.title(label)
        plt.xlim([0,24])
    except Exception, e:
        print "Exception: ", e

# spatial distribution of each label
fig = plt.figure()
for i,label in enumerate(s.labels):
    empty = np.zeros((600,800), np.float64)
    for f in files[label]:
        #face_20160805-111407_xc261_yc341_w167_h167.png
        xc = int(f.split("/")[-1].split("_")[2][2:])
        yc = int(f.split("/")[-1].split("_")[3][2:])
        w = int(f.split("/")[-1].split("_")[4][1:])
        h = int(f.split("/")[-1].split("_")[5].split(".png")[0][1:])
        empty[yc-h/2:yc+h/2, xc-w/2:xc+w/2] += 1
    fig.add_subplot(len(s.labels), 1, i+1)
    plt.imshow(empty)
    plt.colorbar()
    plt.title(label)

plt.show()




