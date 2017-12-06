import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from keras.utils import to_categorical
from datasets.config import config

class DataSet():
    def __init__(self, seq_length, nb_classes):
        self.seq_length = seq_length
        self.class_limit = nb_classes

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning.
        self.data = self.clean_data()

    @staticmethod
    def get_data():
        """Load our data from file."""
        # TODO Path shouldn't be hard coded
        with open(os.path.join('datasets', 'ucf101', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data

    def sample_filter(self, x):
        return int(x[3]) >= self.seq_length and x[1] in self.classes

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        return list(filter(self.sample_filter, self.data))

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = sorted(set([item[1] for item in self.data]))

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert len(label_hot) == len(self.classes)

        return label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        # Two loops, only called twice and data is small.
        train = list(filter(lambda x: x[0] == 'train', self.data))
        test = list(filter(lambda x: x[0] == 'test', self.data))
        return train, test

    def frame_generator(self, batch_size, train_test):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))

        while 1:
            x = []
            y = np.zeros((batch_size, len(self.classes)))
            samples = random.sample(data, batch_size)

            # Generate batch_size samples.
            for i, sample in enumerate(samples):
                # Get and resample frames.
                frames = self.get_frames_for_sample(sample)

                # Get a random start point for this sample and grab seq_length frames
                start = random.randint(0, len(frames) - self.seq_length)
                sampled_frames = frames[start:start + self.seq_length]

                x.append(sampled_frames)
                y[i] = self.get_class_one_hot(sample[1])

            yield x, y

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        # TODO This shouldn't be hard coded
        path = os.path.join('datasets', 'ucf101', sample[0], sample[1], sample[2] + '*jpg')
        images = sorted(glob.glob(path))
        return images

    @staticmethod
    def print_class_from_prediction(predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        # Get the prediction for each label.
        label_predictions = {}
        for i, label in enumerate(data.classes):
            label_predictions[label] = predictions[i]

        # Now sort them.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # And return the top N.
        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
