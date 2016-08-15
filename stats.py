#!/usr/bin/python
'''
display stats regarding the person and the times we saw him/her
'''

from settings import *
import glob
import matplotlib.pyplot as plt
import numpy as np
import datetime

s = Settings()
daynames = [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun"
        ]
folder = "./faces_120/"

hours = {}
days = {}
widths = {}

files = {}
for label in s.labels:
    files[label] = glob.glob(folder + label + "/*.png")
    files[label].sort()
    print "Got %d files for label %s" % (len(files[label]), label)
    for f in files[label]:
        # extract hour
        if f.split("-")[1][:6] != "010101":
            hour = int(f.split("-")[1][:2])
            date = date = f.split("/")[-1].split("_")[1].split("-")[0]
            year = int(date[0:4])
            month = int(date[4:6])
            day = int(date[6:8])
            weekday = str(datetime.datetime(year=year, month=month, day=day).weekday()) # monday:0, sunday: 6

            if label in days.keys():
                if weekday in days[label].keys():
                    days[label][weekday].append(hour)
                else:
                    days[label][weekday] = []
                    days[label][weekday].append(hour)
            else:
                days[label] = {}
                days[label][weekday] = []
                days[label][weekday].append(hour)

            #face_20160604-070415_xc181_yc171_w70_h70.png
            w = f.split("_w")[1].split("_")[0]
            h = f.split("_h")[1].split(".")[0]

            if label not in widths.keys():
                widths[label] = []
            widths[label].append(int(w))


# number of pics per label per hour
fig = plt.figure(figsize=(40,5))
images = []
totalmax = 0.
for i,label in enumerate(s.labels):
    try:
        empty = np.zeros((24,7), np.int32)
        for k,v in days[label].iteritems():
            for hour in v:
                empty[hour,int(k)] += 1
        if np.max(empty) > totalmax and label != 'Negative':
            totalmax = np.max(empty)
        images.append(empty)
    except Exception, e:
        print "Exception: ", e

for i,label in enumerate(s.labels):
    fig.add_subplot(1, len(s.labels), i+1)
    plt.imshow(images[i], interpolation='nearest', vmin=0., vmax=totalmax)
    plt.xlabel('weekday')
    plt.ylabel('time')
    plt.xticks(range(7), daynames, rotation='vertical')
    #plt.colorbar()
    #plt.xticks(range(24))
    #plt.xlim([0,24])
    plt.title(label)


# spatial distribution of each label
fig = plt.figure(figsize=(5,40))
for i,label in enumerate(s.labels):
    empty = np.zeros((600,800), np.float64)
    for f in files[label]:
        if f.split("-")[1][:6] != "010101":
            #face_20160805-111407_xc261_yc341_w167_h167.png
            xc = int(f.split("/")[-1].split("_")[2][2:])
            yc = int(f.split("/")[-1].split("_")[3][2:])
            w = int(f.split("/")[-1].split("_")[4][1:])
            h = int(f.split("/")[-1].split("_")[5].split(".png")[0][1:])
            empty[yc-h/2:yc+h/2, xc-w/2:xc+w/2] += 1

    fig.add_subplot(len(s.labels), 1, i+1)
    plt.imshow(empty)
    #plt.colorbar()
    plt.title(label)


# image sizes of face patches
fig = plt.figure()
plt.suptitle("Width distribution of labels")
for i,label in enumerate(s.labels):
    fig.add_subplot(1, len(s.labels), i+1)
    plt.hist(widths[label])
    plt.xticks(np.arange(-10., max(widths[label]), 20.))
    plt.title(label)
    
plt.show()

