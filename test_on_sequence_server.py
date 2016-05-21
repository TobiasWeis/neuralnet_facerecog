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

from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from SocketServer import ThreadingMixIn


class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            while True:
                try:
                    for f in files:
                        frame = cv2.imread(f)

                        # Do some processing here!

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
