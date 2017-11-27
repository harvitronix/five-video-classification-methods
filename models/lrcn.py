from keras.layers import Dense, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling2D

def model(nb_classes, input_shape):
    """Build a CNN into RNN.
    Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py

    Heavily influenced by VGG-16:
        https://arxiv.org/abs/1409.1556

    Also known as an LRCN:
        https://arxiv.org/pdf/1411.4389.pdf
    """
    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
        activation='relu', padding='same'), input_shape=input_shape))
    model.add(TimeDistributed(Conv2D(32, (3,3),
        kernel_initializer="he_normal", activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(64, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(128, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(128, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(256, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(256, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
    
    model.add(TimeDistributed(Conv2D(512, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(512, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

