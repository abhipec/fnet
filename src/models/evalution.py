"""
Helper file for fine-grained entity type evaluation.
"""

import numpy as np

def invert_dict(dictionary):
    """
    Invert a dict object.
    """
    return {v:k for k, v in dictionary.items()}

def add_label(label, existing_label):
    """
    Generate a leaf node of label hierarchy.
    """
    new_label = None
    if existing_label:
        if existing_label in label:
            new_label = label
        else:
            new_label = existing_label
    else:
        new_label = label
    return new_label

def expand_leaf_to_label(leaf_label):
    """
    Expand a leaf node label into flat structure.
    """
    labels = []
    parts = list(filter(None, leaf_label.split('/')))
    for i, _ in enumerate(parts):
        labels.append('/' + '/'.join(parts[0:i+1]))
    return labels

def hierarchical_prediction(predictions, num_to_label, threshold=0):
    """
    Return hierarchical predictions.
    """

    label_to_num = invert_dict(num_to_label)

    p_labels = []

    for prediction in predictions:
        sorted_p_ids = np.argsort(prediction)[::-1]
        actual_p = None
        not_first = False
        for pid in sorted_p_ids:
            # allow the maximum prediction even if it is less than threshold
            if not_first and prediction[pid] <= threshold:
                break
            not_first = True
            actual_p = add_label(num_to_label[pid], actual_p)
#            print(num_to_label[pid])
        actual_p = expand_leaf_to_label(actual_p)
#        print(actual_p)
        p_labels.append(actual_p)
    new_predictions = np.zeros(predictions.shape, dtype=np.core.numerictypes.float32)
    for i, labels in enumerate(p_labels):
        for label in labels:
            new_predictions[i][label_to_num[label]] = 1
    return new_predictions
