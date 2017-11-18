# Five video classification methods

The five video classification methods:

1. Classify one frame at a time with a ConvNet
1. Extract features from each frame with a ConvNet, passing the sequence to an RNN, in a separate network
1. Use a time-dstirbuted ConvNet, passing the features to an RNN, much like #2 but all in one network (this is the `lrcn` network in the code).
1. Extract features from each frame with a ConvNet and pass the sequence to an MLP
1. Use a 3D convolutional network (has two versions of 3d conv to choose from)

See the accompanying blog post for full details: https://medium.com/@harvitronix/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5

## Requirements

This code requires you have Keras 2 and TensorFlow 1 or greater installed. Please see the `requirements.txt` file. To ensure you're up to date, run:

`pip install -r requirements.txt`

You must also have `ffmpeg` installed in order to extract the video files. If `ffmpeg` isn't in your system path (ie. `which ffmpeg` doesn't return its path, or you're on an OS other than *nix), you'll need to update the path to `ffmpeg` in `data/2_extract_files.py`.

## Getting the data

First, download the dataset from UCF into the `data` folder:

`cd data && wget http://crcv.ucf.edu/data/UCF101/UCF101.rar`

Then extract it with `unrar e UCF101.rar`.

Next, create folders (still in the data folder) with `mkdir train && mkdir test && mkdir sequences && mkdir checkpoints`.

Now you can run the scripts in the data folder to move the videos to the appropriate place, extract their frames and make the CSV file the rest of the code references. You need to run these in order. Example:

`python 1_move_files.py`

`python 2_extract_files.py`

## Extracting features

Before you can run the `lstm` and `mlp`, you need to extract features from the images with the CNN. This is done by running `extract_features.py`. On my Dell with a GeFore 960m GPU, this takes about 8 hours. If you want to limit to just the first N classes, you can set that option in the file.

## Training models

The CNN-only method (method #1 in the blog post) is run from `train_cnn.py`.

The rest of the models are run from `train.py`. There are configuration options you can set in that file to choose which model you want to run.

The models are all defined in `models.py`. Reference that file to see which models you are able to run in `train.py`.

Training logs are saved to CSV and also to TensorBoard files. To see progress while training, run `tensorboard --logdir=data/logs` from the project root folder.

## Demo/Using models

I have not yet implemented a demo where you can pass a video file to a model and get a prediction. Pull requests are welcome if you'd like to help out!

## TODO

- [ ] Add data augmentation to fight overfitting
- [x] Support multiple workers in the data generator for faster training
- [ ] Add a demo script
- [ ] Support other datasets
- [ ] Implement optical flow
- [ ] Implement more complex network architectures, like optical flow/CNN fusion

## UCF101 Citation

Khurram Soomro, Amir Roshan Zamir and Mubarak Shah, UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild., CRCV-TR-12-01, November, 2012. 

