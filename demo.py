"""

Given a video path and a saved model (checkpoint), produce classification
predictions.
if using a model that requires features to be extracted, those
features will be extracted automatically. 

"""
import os.path
import numpy as np
import cv2
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import pandas as pd 
import random
import operator

class Extractor():
    def __init__(self, weights=None):
        self.weights = weights  
        if weights is None:
            base_model = InceptionV3(
                weights='imagenet',
                include_top=True
            )

            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )

        else:

            self.model = load_model(weights)
            self.model.layers.pop()
            self.model.layers.pop()  
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, img):
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        features = self.model.predict(x)
        features = features[0]
        return features

def rescale_list(input_list, size, two_way_rescale = False):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list;if we have
        a list of size 3, return a new list of size five which is filled randomly with
        previous frames.
        """
    if len(input_list) >= size:

        # Get the number to skip between iterations.
        skip = len(input_list) // size
 
        # Build our new output.        
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.        
        return output[:size]

    elif two_way_rescale == True:

        # Build our new output.                
        append_list = []
        for i in range(size - len(input_list)):
            append_list.append(input_list[random.randint(0,len(input_list)-1)])

        # Merge two lists.        
        output = append_list+input_list
        return output


def process_image(image, target_shape):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape
    image = image.load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = image.img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)

    return x

def build_image_sequence(frames):
    """Given a set of frames (filenames), build our sequence."""
    return [process_image(x, image_shape) for x in frames]

def get_sequences(video,seq_length, datatype, imageshape):

    if datatype == 'images':
        # read the video
        videoCapture = cv2.VideoCapture(video)
        # read frames
        success, frame = videoCapture.read()
        sequence = []
        while success:
            image = image.img_to_array(frame)
            frames.append(image)
            success, frame = videoCapture.read()
        # rescale our sequences
        frames = rescale_list(frames, seq_length)
        sequence = build_image_sequence(frames,imageshape)
        return sequence        

    elif datatype == 'features':
        # check if the sequences exists
        path = os.path.join('sequences/' + video + '-' + str(seq_length) + '.npy')
        if os.path.isfile(path):
            print('load existing sequence...')
            return np.load(path)
        else:
            print('generating sequence...')
            videoCapture = cv2.VideoCapture(video)
            success, frame = videoCapture.read()
            sequence = []
            while success:
                frame = image.img_to_array(frame)
                # extract features
                features = extractor.extract(frame)
                sequence.append(features)
                success, frame = videoCapture.read()
            # rescale our sequences
            sequence = rescale_list(sequence, seq_length)
            np.save(path, sequence)     
            return sequence

    else:
        raise ValueError("Invalid data type.")

def print_class_from_prediction(classes, predictions, nb_to_return=5):
    """Given a prediction, print the top classes."""
    # Get the prediction for each label.
    label_predictions = {}
    for i, label in enumerate(classes):
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


def get_classes(class_path,class_limit = None):
    '''
    class_file should look like this
    1 ApplyEyeMakeup
    2 ApplyLipstick
    3 Archery
    4 BabyCrawling
    5 BalanceBeam
    6 BandMarching
    '''
    classes = pd.read_csv(class_path,sep = '',header = None)[1].tolist()
    classes = sorted(classes)
    if class_limit is not None:
        return classes[:class_limit]
    else:
        return classes


def main():
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d.
    model = 'lstm'
    # Must be a weights file.
    saved_model = 'data/checkpoints/lstm-features.026-0.239.hdf5'
    # Sequence length must match the lengh used during training.
    seq_length = 40
    # Limit must match that used during training.
    class_limit = 4

    video_name = 'v_ApplyLipstick_g01_c01'
    model = load_model(saved_model)

    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)

    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    # Extract the sample from the data.
    # seq_length should match the pretrained model
    sample = get_sequences(video_name,seq_length, datatype =  data_type, image_shape = image_shape)

   classes =get_classes('data/ucfTrainTestlist/classInd.txt',class_limit)

    prediction = model.predict(np.expand_dims(sample, axis=0))
    print(prediction)
    data.print_class_from_prediction(classes, np.squeeze(prediction, axis=0))

if __name__ == '__main__':
    main()
