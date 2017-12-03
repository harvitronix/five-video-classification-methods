"""
Given a trained model, evaluate it and generate a confusion matrix.

TODO: This is super hacky and early days. Needs to be generalized and
made to be, you know, useful.
"""
from config import config
if config['CPU_only']:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from itertools import chain
from sklearn import metrics
from datasets.config import config as datasets_config
from datasets.base import get_generators
from models.config import config as models_config
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse

# Replace with dynamic class names.
CLASSES = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth']

def evaluate(model, val_gen):
    # Get a subset of the test data. We do this so we can compare the predictions
    # to the actual, without having to deal with the randomization of the generators.
    # TODO There has to be a better way to do this part
    eval_data = [next(val_gen) for __ in range(config['validation_steps'])]
    # get each sample from each step and batch
    x = []
    y = []
    for row in eval_data:
        for sample in row[0][0]:
            x.append(sample)
        for sample in row[1]:
            y.append(sample)
    eval_x = np.array(x)
    eval_y = np.array(y)
    
    # Now predict with the trained model and compare to the actual.
    predictions = model.predict(eval_x, verbose=1, batch_size=8)
    predictions_index = predictions.argmax(axis=-1)
    actual_index = eval_y.argmax(axis=-1)
    confusion_matrix = metrics.confusion_matrix(actual_index, predictions_index,
                                                labels=list(range(len(CLASSES))))
    print(confusion_matrix)

    # Visualize the confusion matrix.
    df_cm = pd.DataFrame(confusion_matrix, index=[i for i in CLASSES],
                         columns=[i for i in CLASSES])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main(model_name, dataset, model_path):
    paths = models_config['models'][model_name]['paths']
    input_shapes = [x['input_shape'] for x in paths]
    preprocessing_steps = [x['preprocessing'] for x in paths]
    if config['class_limit']:
        nb_classes = config['class_limit']
    else:
        nb_classes = datasets_config['datasets'][dataset]['nb_classes']

    # Get the model.
    model = load_model(model_path)

    # Get the data generators.
    generators = get_generators(dataset, config['sequence_length'], nb_classes,
                                input_shapes, preprocessing_steps,
                                config['batch_size']) 
    __, val_gen = generators
    
    evaluate(model, val_gen)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a classifier.')
    parser.add_argument('model_name', action='store',
                        help='The model to be trained',
                        choices=list(models_config['models'].keys()))
    parser.add_argument('dataset', action='store',
                        help='The dataset to train on',
                        choices=list(datasets_config['datasets'].keys()))
    parser.add_argument('model_path', action='store',
                        help='Saved model to load.')
    args = parser.parse_args()

    main(args.model_name, args.dataset, args.model_path)
