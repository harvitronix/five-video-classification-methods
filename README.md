# Keras Video Classification Playground

This playground aims to make it simple to perform video classification research using Python, Keras and the TensorFlow backend. It is a collection of models, video datasets and pre-processors that you can use to find your video deep learning zen.

The code is roughly split into the following sections:

1. Datasets - we aim to provide easy access to top video datasets
2. Models - we provide a handful of starting models and encourage you to add your own
3. Pre-processors - whether you want raw image frames or optical flow, we aim to offer a shelf full of preprocessors that youc an use on the datasets
4. Train/eval - with your dataset, model and pre-processors in hand, train and evaulate your performance

## Contributing

Why is this section so high up? Well, we really hope that you will contribute your own daatasets, models and pre-processors to the playground. So if you have something to share, make a PR!

## How it works

### Off-the-shelf
The following are included in the playground for you to use with minimal configuration.

### Datasets

### Models

### Preprocessors


### Adding your own
Want to use your own datasets, models, and/or pre-processors? We've made it relatively easy to plug your own into the playground, so you can make use of one or more of the items described above, swapping out your own where you want.

We hope that if you can make any of your research available, you will make a PR and we'll make it available for others.

#### Datasets

#### Models

#### Pre-processors

## Installation & Requirements

TODO: Super quick install and run guide.
TODO: virtualenv/git clone/etc.

This code requires you have Keras 2 and TensorFlow 1 or greater installed. Please see the `requirements.txt` file. To ensure you're up to date, run:

`pip install -r requirements.txt`

You must also have `ffmpeg` installed in order to extract the video files. If `ffmpeg` isn't in your system path (ie. `which ffmpeg` doesn't return its path, or you're on an OS other than *nix), you'll need to update the path to `ffmpeg` in `data/2_extract_files.py`.

## Origins

This project started as a collection of code to support my blog post, [Five video classification methods implemented in Keras and TensorFlow](https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5). If you're looking for that version, see the [releases](https://github.com/harvitronix/five-video-classification-methods/releases/tag/v1.0) section.

## License

MIT, unless otherwise noted.
