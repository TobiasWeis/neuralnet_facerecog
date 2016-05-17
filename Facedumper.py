#!/usr/bin/python
import cv2
import sys
import os
import numpy as np
from datetime import datetime
import threading
import time

class Facedumper(object):
    def __init__(self):
        self.shouldRun = True
        self.exposure = 120
        self.w = 800 
        self.h = 600
        self.device = "/dev/videoc920"

        self.folder = "/home/sarah/scripts/faces_%d" % self.exposure
        if not os.path.isdir(self.folder):
                os.mkdir(self.folder)

        cascPath = "/home/sarah/scripts/haarcascade_frontalface_alt.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath)

        self.video_capture = cv2.VideoCapture(0)

        self.video_capture.set(3, self.w)
        self.video_capture.set(4, self.h)

        os.system("uvcdynctrl -d %s -s\"White Balance Temperature, Auto\" -- 0" % (self.device)) # white balance
        os.system("uvcdynctrl -d %s -s\"White Balance Temperature\" -- 4000" % (self.device)) # white balance
        os.system("uvcdynctrl -d %s -s\"Backlight Compensation\" -- 0" % (self.device)) # backlight compensation
        os.system("uvcdynctrl -d %s -s\"Gain\" -- 190" % (self.device)) # gain
        os.system("uvcdynctrl -d %s -s\"Brightness\" -- 128" % (self.device)) # brightness
        os.system("uvcdynctrl -d %s -s\"Exposure, Auto\" -- 1" % (self.device)) # turn off auto-exposure
        os.system("uvcdynctrl -d %s -s\"Exposure (Absolute)\" -- %d" % (self.device, self.exposure))
        #os.system("uvcdynctrl -d /dev/video0 -s\"Focus, Auto\" -- 0") # turn off auto-focus
        #os.system("uvcdynctrl -d /dev/video0 -s\"Focus (absolute)\" -- 0") # set focus to 0

        #d some frames to flush the buffer
        for i in range(0,30):
            os.system("uvcdynctrl -d %s -s\"Exposure (Absolute)\" -- %d" % (self.device, self.exposure))
            ret, frame = self.video_capture.read()


        self.last_saved = 0
        self.delay = 1 # seconds in between saving frames


    def capture(self):
        print "Starting capture!"
        self.shouldRun = True

        while self.shouldRun:
            os.system("uvcdynctrl -d %s -s\"Exposure (Absolute)\" -- %d" % (self.device, self.exposure))
            # Capture frame-by-frame
            ret, frame = self.video_capture.read()
            # cam is mounted upside down
            frame = cv2.flip(frame,0)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )

            if time.time() - self.last_saved > self.delay:
                for (x, y, w, h) in faces:
                    # save patch
                    d = datetime.now()
                    cv2.imwrite("%s/face_%d%d%d-%02d%02d%02d_xc%d_yc%d_w%d_h%d.png" % (self.folder, d.year, d.month, d.day, d.hour, d.minute, d.second, (x + w/2), (y + h/2), w, h), frame[y:y+h, x:x+w,:])
                    cv2.imwrite("%s/complete_%d%d%d-%02d%02d%02d_xc%d_yc%d_w%d_h%d.png" % (self.folder, d.year, d.month, d.day, d.hour, d.minute, d.second, (x + w/2), (y + h/2), w, h), frame)
                    self.last_saved = time.time()

            cv2.waitKey(35)



