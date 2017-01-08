"""
Evaluation metrics for fnet.
"""
import numpy as np
def f1_score(precision, recall):
    """
    Compute f1 score.
    """
    if precision or recall:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0

def strict(predictions,
           targets):
    """
    Implementation of strict evaluation metric used by Ling et al.
    Since all entities are already identified, in our case P = T.
    Args:
        predictions: A numpy array of shape [batch_size, labels].
        targets: A numpy array of shape [batch_size, labels]
    """
    assert predictions.shape == targets.shape,\
        "Prediction and target shape should be equal."
    ids_that_model_has_predicted = np.sum(predictions, 1) > 0
    P = np.sum(ids_that_model_has_predicted)

    ids_that_have_target = np.sum(targets, 1) > 0
    T = np.sum(ids_that_have_target)

    ids_intersection = ids_that_model_has_predicted & ids_that_have_target

    predictions = predictions[ids_intersection]
    targets = targets[ids_intersection]

    _, L = targets.shape

    precision = np.sum(np.sum(predictions == targets, 1) == L) / P
    recall = np.sum(np.sum(predictions == targets, 1) == L) / T

    return f1_score(precision, recall)

def loose_macro(predictions, targets):
    """
    Implemetation of loose macro evaluation metric used by Ling et al.
    Since all entities are already identified, in our case P = T.
    Args:
        predictions: A numpy array of shape [batch_size, labels].
        targets: A numpy array of shape [batch_size, labels]
    """
    assert predictions.shape == targets.shape,\
        "Prediction and target shape should be equal."
    ids_that_model_has_predicted = np.sum(predictions, 1) > 0
    P = np.sum(ids_that_model_has_predicted)

    p = predictions[ids_that_model_has_predicted]
    t = targets[ids_that_model_has_predicted]
    precision = np.sum(np.sum((p != 0) & (t != 0), 1) / np.sum(p, 1)) / P


    ids_that_have_target = np.sum(targets, 1) > 0
    T = np.sum(ids_that_have_target)

    p = predictions[ids_that_have_target]
    t = targets[ids_that_have_target]

    recall = np.sum(np.sum((p != 0) & (t != 0), 1) / np.sum(t, 1)) / T

    return f1_score(precision, recall)

def loose_micro(predictions, targets):
    """
    Implemetation of loose micro evaluation metric used by Ling et al.
    Since all entities are already identified, in our case P = T.
    Args:
        predictions: A numpy array of shape [batch_size, labels].
        targets: A numpy array of shape [batch_size, labels]
    """
    assert predictions.shape == targets.shape,\
        "Prediction and target shape should be equal."
    ids_that_model_has_predicted = np.sum(predictions, 1) > 0
    P = np.sum(ids_that_model_has_predicted)

    p = predictions[ids_that_model_has_predicted]
    t = targets[ids_that_model_has_predicted]
    precision = np.sum(np.sum((p != 0) & (t != 0), 1)) / np.sum(np.sum(p, 1))

    ids_that_have_target = np.sum(targets, 1) > 0
    T = np.sum(ids_that_have_target)

    p = predictions[ids_that_have_target]
    t = targets[ids_that_have_target]

    recall = np.sum(np.sum((p != 0) & (t != 0), 1)) / np.sum(np.sum(t, 1))
    return f1_score(precision, recall)

def _non_exhaustive_check():
    predictions = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 0]
        ])
    targets = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [1, 1, 0],
        [0, 1, 0]
        ])
    assert np.abs(0.428571 - strict(predictions, targets)) < 1e-5
    assert np.abs(0.618131 - loose_macro(predictions, targets)) < 1e-5
    assert np.abs(0.695652 - loose_micro(predictions, targets)) < 1e-5

if __name__ == '__main__':
    _non_exhaustive_check()
