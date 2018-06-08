"""
Class for managing our data.
"""
import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from processor import process_image
from keras.utils import to_categorical

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():

    def __init__(self, seq_length=40, class_limit=None, image_shape=(224, 224, 3)):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join('data', 'sequences')
        self.max_frames = 300  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning.
        self.data = self.clean_data()

        self.image_shape = image_shape

    @staticmethod
    def get_data():
        """Load our data from file."""
        with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

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
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences_in_memory(self, train_test, data_type):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Loading %d samples into memory for %sing." % (len(data), train_test))

        X, y = [], []
        for row in data:

            if data_type == 'images':
                frames = self.get_frames_for_sample(row)
                frames = self.rescale_list(frames, self.seq_length)

                # Build the image sequence
                sequence = self.build_image_sequence(frames)

            else:
                sequence = self.get_extracted_sequence(data_type, row)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    raise

            X.append(sequence)
            y.append(self.get_class_one_hot(row[1]))

        return np.array(X), np.array(y)

    @threadsafe_generator
    def frame_generator(self, batch_size, train_test, data_type):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample = random.choice(data)

                # Check to see if we've already saved this sequence.
                if data_type is "images":
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames, self.seq_length)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)
                else:
                    # Get the sequence from disk.
                    sequence = self.get_extracted_sequence(data_type, sample)

                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")

                X.append(sequence)
                y.append(self.get_class_one_hot(sample[1]))

            yield np.array(X), np.array(y)
    
    @threadsafe_generator
    def in_memory_generator(self, batch_size, train_test, data_type, augmentation=False, verbose=False):
        """COPIED:
        Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        EDIT: this generator DOES load everything in memory. Other than get_all_sequences_in_memory(),
        this functions performs some data augmentation (DA) on the fly. Since we want to perform DA on a single
        sample with multiple frames at once, we cant use per-frame-DA (i.e. if we randomly rotate a frame,
        we want to rotate every frame within the same sample with the same amount.
        This custom data loader provides such DA,
        yet while being faster than applying regular DA in the non-memory generator
        """
        # Get the right dataset for the generator.
        # a separate test and validation set is currently not included
        # if train_test == 'test':
        #     data = self.split_test()
        # else:
        train, val = self.split_train_val()
        data = train if train_test == 'train' else val

        print("\nCreating %s generator in memory, with %d samples." % (train_test, len(data)))

        X, y = self.get_all_sequences_in_memory(train_test, data_type)

        while 1:

            idx = list(range(X.shape[0]))
            # random.shuffle(idx)

            while idx:

                batch_idx = idx[:batch_size]
                idx = idx[batch_size:]
                X_batch, y_batch = [], []

                # Generate batch_size samples.
                for i in batch_idx:
                    # Reset to be safe.
                    sample = None

                    # get a copy of the original data
                    sample = copy.deepcopy(X[i])
                    label = copy.deepcopy(y[i])

                    if augmentation: sample, label = self.augment(sample, label)

                    X_batch.append(sample)
                    y_batch.append(label)

                yield np.array(X_batch), np.array(y_batch)

    def augment(self, sample, label):
        """
        Performing augmentation on the fly
        each sample is a numpy array with shape: [# frames per sequence, H, W 3]
        """

        # sample info
        H, W, c = sample[0].shape
        seq_len = self.seq_length
        assert len(sample) == seq_len, 'Somehow sequence lenght isn\'t correct!'

        # augmentation hyperparams
        angle     = 20   # maximum absolute (left or right) angle for rotation
        mu        = 0    # mean for guassian noise
        sigma     = 0.1  # std for gaussian noise
        gamma_min = 0.5  # gamma for gamma conversion MAKE RANDOM UP TO 0.5?
        flip_h_ch = 0.2  # chance of applying horizontal flip
        flip_v_ch = 0.2  # chance of applying vertical flip

        # augmentation parameters that need to be constant for a single sample
        rand_rot = np.random.randint(-angle, angle)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rand_rot, 1)

        flip_hor = np.random.uniform() < flip_h_ch
        flip_ver = np.random.uniform() < flip_v_ch

        # loop over each frame only once for efficiency (i.e. not in every of the augmentation functions
        for i, frame in enumerate(sample):

            # apply augmentations
            frame = self.rotate(frame, M, H, W)
            frame = self.gaussian_noise(frame, H, W, c, mu=mu, sigma=sigma)
            frame = self.gamma_conversion(frame, gamma_min=gamma_min)
            if flip_hor: frame = self.flip_horizontal(frame)
            if flip_ver: frame = self.flip_vertical(frame)

            # update the frame in the sample
            sample[i] = frame

        # flip the label in case the images gets horizontally flipped
        if flip_ver: label = np.ones_like(label, dtype=label.dtype) - label

        return sample, label

    @staticmethod
    def rotate(frame, M, H, W):
        '''Applies same/consistent rotation to each frame in a sample'''

        return cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def gaussian_noise(frame, H, W, c, mu=0, sigma=0.1):
        '''Applies different random/guassian noise separately to each frame of sample'''

        # get noise params, okay to be different for each frame in a single sample
        gauss = np.random.normal(mu, sigma, (H, W, c))
        # apply noise
        frame = np.clip(frame + gauss, 0, 1)

        return frame

    @staticmethod
    def flip_vertical(frame):
        '''Applies vertical flipping similarly/consistently over all frame of sample'''

        frame = frame[:, ::-1, :]

        return frame

    @staticmethod
    def flip_horizontal(frame):
        '''Applies horizontal flipping similarly/consistently over all frame of sample'''

        frame = frame[::-1, :, :]

        return frame

    @staticmethod
    def translate(sample, seq_len, H, W, c, mu=0, sigma=0.1):
        # NOT IMPLEMENTED
        return sample

    @staticmethod
    def gamma_conversion(frame, gamma_min=0.75):
        '''Applies gamma conversion separately to each frame of sample'''

        # get conversion params, might be okay to change for each frame?
        gamma = np.random.uniform(gamma_min, 1.)
        gamma = 1 / gamma if np.random.uniform() < 0.5 else gamma

        # apply conversion
        frame = frame ** (1.0 / gamma)

        return frame
    
    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x, self.image_shape) for x in frames]

    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) + \
            '-' + data_type + '.npy')
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_frames_by_filename(self, filename, data_type):
        """Given a filename for one of our samples, return the data
        the model needs to make predictions."""
        # First, find the sample row.
        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError("Couldn't find sample: %s" % filename)

        if data_type == "images":
            # Get and resample frames.
            frames = self.get_frames_for_sample(sample)
            frames = self.rescale_list(frames, self.seq_length)
            # Build the image sequence
            sequence = self.build_image_sequence(frames)
        else:
            # Get the sequence from disk.
            sequence = self.get_extracted_sequence(data_type, sample)

            if sequence is None:
                raise ValueError("Can't find sequence. Did you generate them?")

        return sequence

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = os.path.join('data', sample[0], sample[1])
        filename = sample[2]
        images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
        return images

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split(os.path.sep)
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        assert len(input_list) >= size

        # Get the number to skip between iterations.
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]

    def print_class_from_prediction(self, predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        # Get the prediction for each label.
        label_predictions = {}
        for i, label in enumerate(self.classes):
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
