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
from PIL import Image
import StringIO
import time
from settings import *

# FIXME: test for serving mjpg
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from SocketServer import ThreadingMixIn


global s
s = Settings()

cascPath = "./haarcascade_frontalface_alt.xml"
global faceCascade
faceCascade = cv2.CascadeClassifier(cascPath)

global files
files = glob.glob("./sequence/*.png")
files.sort()

global net
net = s.net
net.load_weights_from(s.net_name)


class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global s
        global faceCascade
        global files
        global net
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            font = cv2.FONT_HERSHEY_SIMPLEX

            while True:
                try:
                    for f in files:
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

                        r,buf = cv2.imencode('.jpg', frame)

                        self.wfile.write("--jpgboundary")
                        self.send_header('Content-type','image/jpeg')
                        self.send_header('Content-length',str(len(buf)))
                        self.end_headers()
                        self.wfile.write(bytearray(buf))
                        #jpg.save(self.wfile,'JPEG')
                        time.sleep(0.05)
                except Exception, e:
                    print "EXCEPTION: ", e
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


server = ThreadedHTTPServer(('localhost',8080),CamHandler)
server.serve_forever()
server.socket.close()
