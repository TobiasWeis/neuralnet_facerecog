from nolearn.lasagne import NeuralNet
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

class Settings():
    def __init__(self):
        self.img_size = 21
        self.labels = ["Tobi", "Mariam", "Other", "Negative"]
        self.net_name = 'facerecognet'
        self.net = self.createnet()

    def createnet(self, imgs=21, nf1=16, nf2=16, fs=5):
        self.img_size = imgs
        print "Settings: set image size to: ", imgs
        self.net = NeuralNet(
            layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
            # input layer
            #input_shape=(None, 3,  self.img_size,  self.img_size), #rgb!
            input_shape=(None, 1, self.img_size,  self.img_size),
            # layer conv2d1
            conv2d1_num_filters=nf1,
            conv2d1_filter_size=(fs, fs),
            conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
            conv2d1_W=lasagne.init.GlorotUniform(),
            # layer maxpool1
            maxpool1_pool_size=(2, 2),
            # layer conv2d2
            conv2d2_num_filters=nf2,
            conv2d2_filter_size=(fs, fs),
            conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
            # layer maxpool2
            maxpool2_pool_size=(2, 2),
            # dropout1
            dropout1_p=0.5,
            # dense
            dense_num_units=251,
            dense_nonlinearity=lasagne.nonlinearities.rectify,
            # dropout2
            dropout2_p=0.5,
            # output
            output_nonlinearity=lasagne.nonlinearities.softmax,
            output_num_units=4,
            # optimization method params
            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.8,
            max_epochs=700,
            verbose=1,
            )


