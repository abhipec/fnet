"""
Case wise analysis of results.
Usage:
    main_our [options]
    main_our -h | --help

Options:
    -h, --help                      Print this.
    --result_file=<path>            Result file path
    --json_file=<path>              Json file path.
    --all_labels_file=<path>
    --dataset=<name>
    --remove_pronominals
"""

import json
import csv
from docopt import docopt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
#pylint: disable=import-error
from models.metrics import strict, loose_macro, loose_micro


def invert_dict(dictionary):
    """
    Invert a dict object.
    """
    return {v:k for k, v in dictionary.items()}

#pylint: disable=too-many-locals
def class_wise_f1_score(prediction,
                        target,
                        num_to_label,
                        sort_id=3,
                        filename=None):
    """
    Compute class wise f1 score.

    Args:
        prediction: A numpy array of shape [batch_size, number_of_labels].
        target: A numpy array of shape [batch_size, number_of_labels].
        num_to_label: A dictionary with index to label name mapping.
        sord_id: Index on which sorting is done. [P, R, F1, S].
        filename: If filename is specified, write csv with sorted output.
    """
    assert prediction.shape == target.shape, \
        "Prediction and target shape should match."
    assert sort_id < 4 and sort_id > -1, \
        "Sort id should be from 0,1,2,3"
    _, number_of_labels = prediction.shape
    output = {}
    for i in range(number_of_labels):
        precision, recall, f_score, _ = precision_recall_fscore_support(target[:, i],
                                                                        prediction[:, i],
                                                                        average='binary')
        output[num_to_label[i]] = [precision, recall, f_score, np.sum(target[:, i]), i]
    # sorting based on score
    sorted_out = sorted(output.items(), key=lambda e: e[1][sort_id], reverse=True)
    if filename:
        # dump score in a file
        with open(filename, 'w') as file_p:
            writer = csv.writer(file_p)
            for key, value in sorted_out:
                writer.writerow([key] + value)
    return output

def generate_labels_to_numbers(filename):
    """
    Generate label to number dictionary.
    """
    with open(filename, 'r') as file_p:
        label_list = file_p.read().split('\n')
        num_to_label = dict(zip(label_list, range(len(label_list))))
        return num_to_label

def create_prediction_dictionary(filename, ltn):
    """
    Convert prediction to a dictionary.
    """
    prediction_dictionary = {}
    with open(filename) as file_p:
        for row in filter(None, file_p.read().split('\n')):
            mention, labels = row.split('\t')
            np_labels = np.zeros(len(ltn))
            for label in filter(None,labels.split(',')):
                np_labels[ltn[label]] = 1
            prediction_dictionary[mention] = np_labels
    return prediction_dictionary

def create_target_dictionary(filename, ltn, predicted_keys, remove_pronominals):
    """
    Convert targets to a dictionary.
    """
    target_dictionary = {}
    with open(filename, 'r', encoding='utf-8') as file_p:
        for row in file_p:
            json_data = json.loads(row)
            pos = json_data['pos']
            for mention in json_data['mentions']:
                uid = '_'.join([json_data['fileid'],
                                str(json_data['senid']),
                                str(mention['start']),
                                str(mention['end'])])
                poss = pos[mention['start']:mention['end']]
                if remove_pronominals:
                    # if all tokens are pronouns, continue to next mention
                    to_add = False
                    for tag in poss:
                        if tag not in ['PRP', 'PRP$', 'WP', 'WP$']:
                            to_add = True
                    if not to_add:
                        continue
                if uid in predicted_keys:
                    np_labels = np.zeros(len(ltn))
                    for label in mention['labels']:
                        np_labels[ltn[label]] = 1
                    target_dictionary[uid] = np_labels
    return target_dictionary

def dictionary_to_np(prediction_dictionary, target_dictionary, ltn):
    """
    Convert prediction and target dictionaries to numpy arrays.
    """
#    assert len(prediction_dictionary) == len(target_dictionary),\
#        "Predictions and targets must be of same dimension."

    np_predictions = np.zeros((len(target_dictionary), len(ltn)))
    np_targets = np.zeros((len(target_dictionary), len(ltn)))
    count = 0
    for uid in target_dictionary:
        np_predictions[count, :] = prediction_dictionary[uid]
        np_targets[count, :] = target_dictionary[uid]
        count += 1
    return np_predictions, np_targets

#pylint: disable=invalid-name
if __name__ == '__main__':
    arguments = docopt(__doc__)
    label_to_num = generate_labels_to_numbers(arguments['--all_labels_file'])
    predictions = create_prediction_dictionary(arguments['--result_file'], label_to_num)
    targets = create_target_dictionary(arguments['--json_file'],
                                       label_to_num,
                                       set(predictions.keys()),
                                       arguments['--remove_pronominals'])
    predictions, targets = dictionary_to_np(predictions, targets, label_to_num)
    print(predictions.shape)
    print(strict(predictions, targets),
          loose_macro(predictions, targets),
          loose_micro(predictions, targets))
    # stop after prinitng final scores, if pronominals were removed.
    if arguments['--remove_pronominals']:
        exit(1)

    class_wise_f1_score(predictions,
                        targets,
                        invert_dict(label_to_num),
                        filename='class_wise_score_' + arguments['--dataset'] + '.csv')
