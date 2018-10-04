import numpy as np


def sensitivity(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + np.epsilon())

def specificity(y_true, y_pred):
    true_negatives = np.sum(np.round(np.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = np.sum(np.round(np.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + np.epsilon())

def _recall(y_true, y_pred):
        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + np.epsilon())
        return recall

def _precision(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + np.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = _precision(y_true, y_pred)
    recall = _recall(y_true, y_pred)
    return 2 * ((precision*recall)/(precision+recall)) 
