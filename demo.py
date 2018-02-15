"""
Given a video path and a saved model (checkpoint), produce classification
predictions.

Note that if using a model that requires features to be extracted, those
features must be extracted first.

Note also that this is a rushed demo script to help a few people who have
requested it and so is quite "rough". :)
"""
from keras.models import load_model
from data import DataSet
import numpy as np

def predict(data_type, seq_length, saved_model, image_shape, video_name, class_limit):
    model = load_model(saved_model)

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(seq_length=seq_length, class_limit=class_limit)
    else:
        data = DataSet(seq_length=seq_length, image_shape=image_shape,
            class_limit=class_limit)
    
    # Extract the sample from the data.
    sample = data.get_frames_by_filename(video_name, data_type)

    # Predict!
    prediction = model.predict(np.expand_dims(sample, axis=0))
    print(prediction)
    data.print_class_from_prediction(np.squeeze(prediction, axis=0))

def main():
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d.
    model = 'lstm'
    # Must be a weights file.
    saved_model = 'data/checkpoints/lstm-features.026-0.239.hdf5'
    # Sequence length must match the lengh used during training.
    seq_length = 40
    # Limit must match that used during training.
    class_limit = 4

    # Demo file. Must already be extracted & features generated (if model requires)
    # Do not include the extension.
    # Assumes it's in data/[train|test]/
    # It also must be part of the train/test data.
    # TODO Make this way more useful. It should take in the path to
    # an actual video file, extract frames, generate sequences, etc.
    #video_name = 'v_Archery_g04_c02'
    video_name = 'v_ApplyLipstick_g01_c01'

    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    predict(data_type, seq_length, saved_model, image_shape, video_name, class_limit)

if __name__ == '__main__':
    main()
