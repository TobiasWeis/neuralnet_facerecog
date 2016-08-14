#!/usr/bin/python
import cv2
import sys
import os
import numpy as np
from datetime import datetime
import threading
import time

import cv2
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import settings
import settings_hotornot

import threading
import shelve

class Facedumper(threading.Thread):
    def __init__(self):
        super(Facedumper, self).__init__()
        self.stoprequest = threading.Event()

        self.last_saved = 0
        self.delay = .5 # seconds in between saving frames

        self.memory = shelve.open("memory", writeback=True)
        self.switch = {}

        self.shouldRun = True
        self.exposure = 120
        self.w = 800 
        self.h = 600
        self.device = "/dev/video0"
        self.s = settings.Settings()
        self.net = self.s.net
        self.net.load_weights_from("/home/sarah/scripts/" + self.s.net_name)

        self.s_hotornot = settings_hotornot.Settings()
        self.net_hotornot = self.s_hotornot.net
        self.net_hotornot.load_weights_from("/home/sarah/scripts/" + self.s_hotornot.net_name)

        self.outfile = "/var/www/smartmirror2/assets/faceimg.png"

        self.folder = "/home/sarah/scripts/faces_%d" % self.exposure
        if not os.path.isdir(self.folder):
                os.mkdir(self.folder)

        cascPath = "/home/sarah/scripts/haarcascade_frontalface_alt.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath)

        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            print "#################################################"
            print " Could not open camera !"
            print "#################################################"
            return

        self.video_capture.set(3, self.w)
        self.video_capture.set(4, self.h)

        '''
        # FIXME: these sometimes cause problems and driver shutdowns

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
        '''



    def __del__(self):
        self.memory.close()
        self.video_capture.release()

    def join(self, timeout=None):
        self.stoprequest.set()
        super(Facedumper, self).join(timeout)
        self.video_capture.release()

    def run(self):
        print "Starting capture!"
        self.shouldRun = True

        while not self.stoprequest.isSet():
            '''
            os.system("uvcdynctrl -d %s -s\"Exposure (Absolute)\" -- %d" % (self.device, self.exposure))
            '''
            # Capture frame-by-frame
            ret, frame = self.video_capture.read()
            # cam is mounted upside down
            frame = cv2.flip(frame,0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            try:
                if time.time() - self.last_saved > self.delay:

                    faces = self.faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(50, 50),
                        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                    )

                    for (x, y, w, h) in faces:
                        # first, save patches to file
                        d = datetime.now()
                        patch = frame[y:y+h, x:x+w,:]
                        cv2.imwrite("%s/face_%d%02d%02d-%02d%02d%02d%03d_xc%d_yc%d_w%d_h%d.png" % (self.folder, d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond/1000, (x + w/2), (y + h/2), w, h), patch)
                        cv2.imwrite("%s/complete_%d%02d%02d-%02d%02d%02d%03d_xc%d_yc%d_w%d_h%d.png" % (self.folder, d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond/1000, (x + w/2), (y + h/2), w, h), frame)
                        self.last_saved = time.time()

                    # now draw the boxes for visualization
                    for (x, y, w, h) in faces:
                        patch = frame[y:y+h, x:x+w,:]
                        # classify, draw rectangle with name,
                        # save to file for the smartmirror to display it
                        if self.s.net.input_shape[1] == 3:
                            img = cv2.resize(patch / 255., (self.s.img_size, self.s.img_size))
                            img = img.transpose(2,0,1).reshape(3, self.s.img_size, self.s.img_size)
                            pred = self.net.predict(np.array([img.astype(np.float32)]))
                        else:
                            img = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) / 255., (self.s.img_size, self.s.img_size))
                            pred = self.net.predict(np.array([[img.astype(np.float32)]]))
                            pred_hotornot = self.net_hotornot.predict(np.array([[img.astype(np.float32)]]))


                        # check if there went some time by since we last saw that person.
                        # if so, display a greeting!

                        if self.memory.has_key(self.s.labels[pred[0]]):
                            print "Had key: ", self.s.labels[pred[0]]
                            print "Allkeys: ", self.memory.keys()

                            if time.time() - self.memory[self.s.labels[pred[0]]]["lastseen"] > 5*60:
                                self.switch[self.s.labels[pred[0]]] = time.time()
                            # save the last-seen timestamp for this person
                            self.memory[self.s.labels[pred[0]]]["lastseen"] = time.time()
                        else:
                            self.memory[self.s.labels[pred[0]]] = {}
                            self.memory[self.s.labels[pred[0]]]["lastseen"] = time.time()
                        self.memory.sync()

                        if pred[0] == 0: # Tobi
                            col = (255,0,0)
                        elif pred[0] == 1: # Mariam
                            col = (255,0,255)
                        elif pred[0] == 2: # Other
                            col = (128,128,128)
                        elif pred[0] == 3:
                            col = (0,0,255)                                                                                            
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.rectangle(frame, (x,y), (x+w,y+h),col,4)
                        cv2.putText(frame,self.s.labels[pred[0]],(x,y-10), font, 1,col,3)

                        # display a greeting for 10 seconds
                        if self.switch.has_key(self.s.labels[pred[0]]):
                            if time.time() - self.switch[self.s.labels[pred[0]]] < 10:
                                cv2.putText(frame, "LONG TIME NO SEE, %s" % (self.s.labels[pred[0]]),(10,50), font, 1, col,3)

                        if pred_hotornot[0] == 0: #Hot
                            cv2.putText(frame,self.s_notornet.labels[pred_hotornot[0]],(x,y-20), font, 1, (0,200,200),3)
                            cv2.rectangle(frame, (x-10,y-10), (x+w+10,y+h+10),(0,200,200),8)


                        cv2.imwrite(self.outfile, frame)
                        print "Written to: ", self.outfile

                cv2.waitKey(35)
            except Exception,e:
                print "Exception: ", e
                pass



