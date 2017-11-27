from keras.applications.vgg16 import VGG16
from keras.layers import LSTM, Dense, Input, TimeDistributed
from keras.models import Model

def model(nb_classes, input_shape):
    """Build a CNN into RNN using a pretrained CNN like VGG, but
    time-distributing it so we can apply it to many frames in a sequence.

    Uses VGG-16:
        https://arxiv.org/abs/1409.1556

    This architecture is also known as an LRCN:
        https://arxiv.org/pdf/1411.4389.pdf
    """
    # Get a pre-trained CNN.
    cnn = VGG16(weights='imagenet', include_top=False, pooling='avg')

    # Don't retrain the VGG
    cnn.trainable = False

    net_input = Input(shape=(None,) +  input_shape, name='net_input')

    # Distribute the CNN over time.
    x = TimeDistributed(cnn)(net_input)

    # Add the LSTM.
    x = LSTM(512, dropout=0.5)(x)

    predictions = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=net_input, outputs=predictions)

    return model

