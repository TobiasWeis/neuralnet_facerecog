#!/bin/bash

# get the new image files from the mirror, delete them on the mirror
scp sarah@192.168.0.191:~/scripts/faces_120/face_*.png ./faces_120/ && ssh sarah@192.168.0.191 'rm /home/sarah/scripts/faces_120/face_*png' 
#&& ssh sarah@192.168.0.191 'rm /home/sarah/scripts/faces_120/complete*.png'

scp sarah@192.168.0.191:~/scripts/faces_120/complete_*png ./faces_120/ && ssh sarah@192.168.0.191 'rm /home/sarah/scripts/faces_120/complete_*png' 

# classifiy them into preliminary folders
./NN_classify_move.py
