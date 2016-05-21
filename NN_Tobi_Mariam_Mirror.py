#!/usr/bin/python
import os
import theano

#import lasagne
#from lasagne import layers
#from lasagne.updates import nesterov_momentum

#from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from settings import *
from utils import *

s = Settings()

_train = True # perform training?
_test = True  # perform teting?

if s.net.input_shape[1] == 3:
    print "Loading rgb training data"
    X_train,y_train,X_test,y_test  = load_faces(s)
else:
    print "Loading grayscale training data"
    X_train,y_train,X_test,y_test  = load_faces(s, rgb=False)

# FIXME: is it possible to do incremental training?

print "# Got the data"

net1 = s.net

print "# Got the net structure"

if _train:
    print "# Training"
    nn = net1.fit(X_train, y_train)

    print "# Saving weights"
    net1.save_weights_to(s.net_name)
if _test:
    print "# Loading weights"
    net1.load_weights_from(s.net_name)

# evaluate
print "# Evaluating"
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
predictions = net1.predict(X_test)

####################### OUTPUT to shell and to file for later
filename = "experiment_log.txt"
target = open(filename, 'a+')
target.write("------------------------------\n")
from nolearn.lasagne import PrintLayerInfo
pli = PrintLayerInfo()
net1.verbose = 3
layer_info, legend = pli._get_layer_info_conv(net1)
target.write(layer_info)
target.write(classification_report(y_test, predictions))
target.write(accuracy_score(y_test, predictions))

print layer_info
print classification_report(y_test, predictions)
print accuracy_score(y_test, predictions)

