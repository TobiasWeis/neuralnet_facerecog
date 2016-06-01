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
for label in s.labels:
    files = glob.glob(folder + label + "/*.png")
    print "Got %d files for label %s" % (len(files), label)
    for f in files:
        # extract hour
        # face_2016429-161616_xc510_yc271_w178_h178.png
        hour = int(f.split("-")[1][:2])
        if label in hours:
            hours[label].append(hour)
        else:
            hours[label] = []
            hours[label].append(hour)

fig = plt.figure()
for i,label in enumerate(s.labels):
    fig.add_subplot(len(s.labels), 1, i+1)
    plt.hist(np.array(hours[label]), bins=range(0,24 + 1, 1))
    plt.xticks(range(24))
    plt.title(label)
    plt.xlim([0,24])

plt.show()




