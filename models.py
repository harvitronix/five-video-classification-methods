"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Dense, Flatten, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Convolution2D, MaxPooling3D, Convolution3D
from collections import deque

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048):
        """
        `model` = one of:
            lstm
            crnn
            mlp
            conv_3d
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lstm':
            print("Loading Deep LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        elif model == 'crnn':
            print("Loading CRNN model.")
            self.input_shape = (seq_length, 224, 224, 3)
            self.model = self.crnn()
        elif model == 'mlp':
            print("Loading simple MLP.")
            self.input_shape = features_length * seq_length
            self.model = self.mlp()
        elif model == 'conv_3d':
            print("Loading Conv3D")
            self.input_shape = (seq_length, 112, 112, 3)
            self.model = self.conv_3d()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=1e-6)  # aggressively small learning rate
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=self.input_shape,
                       dropout_W=0.4, dropout_U=0.4))
        model.add(LSTM(128, return_sequences=True, dropout_W=0.4, dropout_U=0.4))
        model.add(LSTM(128, return_sequences=True, dropout_W=0.4, dropout_U=0.4))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def crnn(self):
        """Build a CNN into RNN.
        Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py
        """
        model = Sequential()
        model.add(TimeDistributed(Convolution2D(24, 5, 5,
            init="he_normal",
            activation='relu',
            subsample=(5, 4),
            border_mode='valid'), input_shape=self.input_shape))
        model.add(TimeDistributed(Convolution2D(32, 5, 5,
            init="he_normal",
            activation='relu',
            subsample=(3, 2),
            border_mode='valid')))
        model.add(TimeDistributed(Convolution2D(48, 3, 3,
            init="he_normal",
            activation='relu',
            subsample=(1, 2),
            border_mode='valid')))
        model.add(TimeDistributed(Convolution2D(64, 3, 3,
            init="he_normal",
            activation='relu',
            border_mode='valid')))
        model.add(TimeDistributed(Convolution2D(128, 3, 3,
            init="he_normal",
            activation='relu',
            subsample=(1, 2),
            border_mode='valid')))
        model.add(TimeDistributed(Flatten()))
        model.add(Dropout(0.2))
        model.add(GRU(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(128, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def mlp(self):
        """Build a simple MLP."""
        # Model.
        model = Sequential()
        model.add(Dense(512, input_dim=self.input_shape))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(Dropout(0.4))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def conv_3d(self):
        """
        Build a 3D convolutional network, C3D.

        Based on the paper:
            https://arxiv.org/pdf/1412.0767.pdf

        As implemented in Keras by:
            https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2

        Note that this requires a lot of memory to run.
        """
        model = Sequential()
        # 1st layer group
        model.add(Convolution3D(64, 3, 3, 3, activation='relu', 
                                border_mode='same', name='conv1',
                                subsample=(1, 1, 1), 
                                input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                               border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv2',
                                subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                               border_mode='valid', name='pool2'))
        # 3rd layer group
        model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3a',
                                subsample=(1, 1, 1)))
        model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                                border_mode='same', name='conv3b',
                                subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                               border_mode='valid', name='pool3'))
        # 4th layer group
        model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                                border_mode='same', name='conv4a',
                                subsample=(1, 1, 1)))
        model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                                border_mode='same', name='conv4b',
                                subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                               border_mode='valid', name='pool4'))
        # 5th layer group
        model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                                border_mode='same', name='conv5a',
                                subsample=(1, 1, 1)))
        model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                                border_mode='same', name='conv5b',
                                subsample=(1, 1, 1)))
        model.add(ZeroPadding3D(padding=(0, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                               border_mode='valid', name='pool5'))
        model.add(Flatten())
        # FC layers group
        model.add(Dense(4096, activation='relu', name='fc6'))
        model.add(Dropout(.5))
        model.add(Dense(4096, activation='relu', name='fc7'))
        model.add(Dropout(.5))
        model.add(Dense(487, activation='softmax', name='fc8'))
        return model

