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
import glob
from settings import *

# FIXME: test for serving mjpg
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from SocketServer import ThreadingMixIn

global frame
frame = None

class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global frame
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    if frame != None:
                        frame = cv2.imread("")
                        imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                        jpg = Image.fromarray(imgRGB)
                        tmpFile = StringIO.StringIO()
                        jpg.save(tmpFile,'JPEG')
                        self.wfile.write("--jpgboundary")
                        self.send_header('Content-type','image/jpeg')
                        self.send_header('Content-length',str(tmpFile.len))
                        self.end_headers()
                        jpg.save(self.wfile,'JPEG')
                        time.sleep(0.05)
                except KeyboardInterrupt:
                        break
                return

                if self.path.endswith('.html'):
                    self.send_response(200)
                    self.send_header('Content-type','text/html')
                    self.end_headers()
                    self.wfile.write('<html><head></head><body>')
                    self.wfile.write('<img src="http://127.0.0.1:8080/cam.mjpg"/>')
                    self.wfile.write('</body></html>')
                    return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """ blablabla """

s = Settings()

cascPath = "./haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

files = glob.glob("./sequence/*.png")
files.sort()

net = s.net
net.load_weights_from(s.net_name)

font = cv2.FONT_HERSHEY_SIMPLEX

server = ThreadedHTTPServer(('localhost',8080),CamHandler)
#from multiprocessing import Process, Array
#p = Process(target=server.serve_forever)
from threading import Thread
p = Thread(target=server.serve_forever)
p.start()

for f in files:
    #print "Opening ", f
    frame = cv2.imread(f)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # use cascade to detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        patch = frame[y:y+h, x:x+w,:]
        # use neural net to classify face
        # transform to right format
        img = cv2.resize(patch / 255., (s.img_size, s.img_size))
        img = img.transpose(2,0,1).reshape(3, s.img_size, s.img_size)

        pred = net.predict(np.array([img.astype(np.float32)]))

        if pred[0] == 0: # Tobi
            col = (255,0,0)
        elif pred[0] == 1: # Mariam
            col = (255,0,255)
        elif pred[0] == 2: # Other
            col = (128,128,128)
        elif pred[0] == 3:
            col = (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h),col,2)
        cv2.putText(frame,s.labels[pred[0]],(x,y-10), font, 1,col,2)


    cv2.imshow("frame", frame)
    cv2.waitKey(10)

p.join()
server.socket.close()
