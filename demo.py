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
import sys, os, json
from yottato.yottato import yottato as yto

def predict(data_type, seq_length, saved_model, image_shape, video_name, class_limit, config):

    model = load_model(saved_model)

    feature_file_path= config.featureFileName
    work_dir = config.workDir
    classlist= config.classes    
    
    # Get the data and process it.
    if image_shape is None:
        data = DataSet(seq_length=seq_length, class_limit=class_limit,
            feature_file_path = feature_file_path,
            repo_dir = config.repoDir,
            work_dir=work_dir, classlist=classlist)
    else:
        data = DataSet(seq_length=seq_length, image_shape=image_shape,
            class_limit=class_limit,
            feature_file_path = feature_file_path,
            repo_dir = config.repoDir,
            work_dir=work_dir, classlist=classlist)
    
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

    #read config file
    if len(sys.argv) > 2:
        configfile = sys.argv[1]
        saved_model = sys.argv[2]
    else:
        print ("Usage: script <fullpath to config.json> <fullpath to HDF5 stored model>")
        sys.exit(0)
        
    yto_config = yto(configfile)
    model = yto_config.videoAlgorithm
    seq_length = yto_config.videoSeqLength

    # Demo file. Must already be extracted & features generated (if model requires)
    # Do not include the extension.
    # Assumes it's in data/[train|test]/
    # It also must be part of the train/test data.
    # TODO Make this way more useful. It should take in the path to
    # an actual video file, extract frames, generate sequences, etc.
    #video_name = 'v_Archery_g04_c02'
    video_name = 'v_ApplyLipstick_g01_c01'

    video_name = os.path.normpath(video_name)
    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    predict(data_type, seq_length, saved_model, image_shape, video_name, class_limit, yto_config)

if __name__ == '__main__':
    main()
