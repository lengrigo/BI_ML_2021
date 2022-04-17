import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_true[i] == 0:
            true_negative += 1
        elif y_pred[i] == y_true[i] == 1:
            true_positive += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            false_negative += 1
        else:
            false_positive += 1

    precision = true_positive
    if true_positive+false_positive != 0:
        precision = true_positive / (true_positive+false_positive)

    recall = true_positive
    if true_positive+false_negative != 0:
        recall = true_positive / (true_positive+false_negative)

    f1 = 0
    if precision+recall != 0:
        f1 = 2*precision*recall / (precision+recall)

    accuracy = (true_positive+true_negative) / (y_pred.shape[0])
    return precision, recall, f1, accuracy

def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    count = 0
    for i in range(y_true.shape[0]):
        if y_pred[i] == y_true[i]:
            count += 1
    accuracy = count / y_true.shape[0]
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    r_squared_result = 1 - (np.sum(np.square(y_true - y_pred) / np.sum(np.square(y_true - np.mean(y_true)))))
    return r_squared_result


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse_result = (1 / y_pred.shape[0]) * np.sum(np.square(y_true - y_pred))
    return mse_result


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae_result = (1 / y_pred.shape[0]) * np.sum(np.abs(y_true - y_pred))
    return mae_result
    