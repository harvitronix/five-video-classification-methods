"""
Base function for retrieving and compiling a model.
"""
import models.c3d as c3d
import models.vgg_rnn as vgg_rnn
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, SGD

def get_model(nb_classes, model_name, seq_length, optimizer,
              learning_rate, input_shapes, verbose):
    """
    `model` = one of:
        lstm
        lrcn
        mlp
        conv_3d
        c3d
    `nb_classes` = the number of classes to predict
    `seq_length` = the length of our video sequences (ie 40)
    `features_length` = the length of the feature vector when we're passing
        pre-extracted features (for example, from a CNN)
    """
    # Only use top k if there's a need.
    metrics = ['accuracy']
    if nb_classes >= 10:
        metrics.append('top_k_categorical_accuracy')

    if model_name == 'c3d':
        model = c3d.model(nb_classes, (seq_length,) + input_shapes[0])
    elif model_name == 'vgg_rnn':
        model = vgg_rnn.model(nb_classes, input_shapes[0])
    else:
        raise ValueError("Unknown network name.")

    if optimizer == 'adam':
        optimizer = Adam(lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = SGD(lr=learning_rate)
    else:
        raise ValueError("Unknown optimizer.")

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                       metrics=metrics)

    if verbose:
        print(model.summary())

    return model

