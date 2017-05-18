"""
Train an LRCN on a stateful LSTM, through a larger CNN.

This requires we pass one image at a time rather than a sequence. Since
we're using a stateful RNN here, we have to reset state at the end of
each sequence. So essentially, we're manually doing what we can automatically
do when passing a sequence to an LSTM. We do this so that we can pass
images through a CNN and backprop all the way from the LSTM back up
through the CNN. We attempted to do this with our TimeDistributed LRCN,
but we ran into memory and transfer learning contraints.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time
import numpy as np

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False):

    # Set variables.
    nb_epoch = 100000
    batch_size = 1

    data = DataSet(
        seq_length=seq_length,
        class_limit=class_limit,
        image_shape=image_shape
    )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Get data.
    X, y = data.get_all_sequences_in_memory('train', data_type, False)
    X_test, y_test = data.get_all_sequences_in_memory('test', data_type, False)

    mean_loss = []
    mean_acc = []

    for epoch in range(nb_epoch):
        # Loop through each sample.
        for i in range(len(X)):

            # Loop through each item in the sequence.
            for j in range(seq_length):

                loss, acc = rm.model.train_on_batch(
                    X[i][j],
                    y[i]
                )

                mean_loss.append(loss)
                mean_acc.append(acc)

                if len(mean_loss) > 100:
                    mean_loss.pop(0)
                if len(mean_acc) > 100:
                    mean_acc.pop(0)

            rm.model.reset_states() 

        print("Mean loss: %.5f" % np.mean(mean_loss))
        print("Mean accuracy: %.5f" % np.mean(mean_acc))


def main():
    """These are the main training settings. Set each before running
    this file."""
    model = 'stateful_lrcn'  # see `models.py` for more
    saved_model = None  # None or weights file
    class_limit = 2  # int, can be 1-101 or None
    seq_length = 40
    load_to_memory = True  # pre-load the sequences into memory

    data_type = 'images'
    image_shape = (150, 150, 3)

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory)

if __name__ == '__main__':
    main()
