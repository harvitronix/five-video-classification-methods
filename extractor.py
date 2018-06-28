from keras.preprocessing import image
from keras.applications import inception_v3, nasnet
from keras.models import Model, load_model
from keras.utils import multi_gpu_model

import numpy as np

class Extractor():
    def __init__(self, weights=None, cnn_model_type='nasnet', n_gpu=1):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            if cnn_model_type == 'InceptionV3':
                self.model = inception_v3.InceptionV3(
                    weights='imagenet',pooling='avg',
                    include_top=False
                )
            elif cnn_model_type == 'nasnet':
                base_model = nasnet.NASNetLarge(
                    weights='imagenet',
                    include_top=True
                )
                # issue https://github.com/keras-team/keras/issues/10109
                self.model = Model(
                    inputs=base_model.input,
                    outputs=base_model.get_layer('global_average_pooling2d_1').output
                )

        else:
            # Load the model first.
            self.model = load_model(weights)
            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

        if n_gpu>1:
            self.model = multi_gpu_model(self.model,n_gpu)

    def extract(self, image_path, cnn_model_type='nasnet'):
        if cnn_model_type== 'InceptionV3':
            target_size = (299, 299)
        elif  cnn_model_type== 'nasnet':
            target_size = (331, 331)

        img = image.load_img(image_path, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        if cnn_model_type == 'InceptionV3':
            x = inception_v3.preprocess_input(x)
        elif cnn_model_type == 'nasnet':
            x = nasnet.preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        features = features[0]

        return features


    def extract_batch(self, image_path_list, cnn_model_type='InceptionV3'):
        if cnn_model_type== 'InceptionV3':
            target_size = (299, 299,3)
            # feature_size = 2048

        elif  cnn_model_type== 'nasnet':
            target_size = (331, 331,3)
            # feature_size = 4032

        batch_size = len(image_path_list)

        X = np.zeros((batch_size,) + target_size )

        for img_idx, image_path in enumerate(image_path_list):
            img = image.load_img(image_path, target_size=target_size[0:2])
            array = image.img_to_array(img)
            X[img_idx] = array
        # x = np.expand_dims(x, axis=0)

        if cnn_model_type == 'InceptionV3':
            X = inception_v3.preprocess_input(X)
        elif cnn_model_type == 'nasnet':
            X = nasnet.preprocess_input(X)

        # Get the prediction.
        features_batch = self.model.predict(X)

        return features_batch
