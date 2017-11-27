from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling3D, Conv3D

def model(nb_classes, input_shape):
    """
    Build a 3D convolutional network, based loosely on C3D.
        https://arxiv.org/pdf/1412.0767.pdf
    """
    # Model.
    model = Sequential()
    model.add(Conv3D(
        32, (3,3,3), activation='relu', input_shape=input_shape, border_mode='same'
    ))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid'))
    model.add(Conv3D(64, (3,3,3), activation='relu', border_mode='same'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(2, 2, 2), border_mode='valid'))
    model.add(Conv3D(128, (3,3,3), activation='relu', border_mode='same'))
    model.add(Conv3D(128, (3,3,3), activation='relu', border_mode='same'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(2, 2, 2), border_mode='valid'))
    model.add(Conv3D(256, (2,2,2), activation='relu', border_mode='same'))
    model.add(Conv3D(256, (2,2,2), activation='relu', border_mode='same'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(2, 2, 2), border_mode='valid'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

