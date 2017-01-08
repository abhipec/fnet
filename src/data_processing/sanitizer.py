"""
Removes entity mentions of zero length from mentions
that survived all pre-processing stages of AFET.

Divide sanitized mentions into train/dev/test split.
"""

import sys
import json
import random
from datetime import datetime

# Test start number according to mention_type_test.txt
TSN = {}
TSN['BBN'] = 86078
TSN['OntoNotes'] = 219634
TSN['Wiki'] = 2677780


def generate_sanitized_mentions(dataset, data_directory, dev_percentage, sanitized_directory):
    """
    Generate and save sanitized mention set.
    """
    sanitized_mentions = {}
    sanitized_mentions['train'] = []
    sanitized_mentions['test'] = []
    sanitized_mentions['dev'] = []
    with open(data_directory + dataset + '/mention.txt', 'r') as file_p:
        for row in file_p:
            if '-1' not in row:
                parts = row.split('\t')
                if int(parts[1]) >= TSN[dataset]:
                    sanitized_mentions['test'].append(parts[0])
                else:
                    sanitized_mentions['train'].append(parts[0])
    dev_choices = len(sanitized_mentions['test']) // dev_percentage
#    time = datetime.now().microsecond
    # hard coded seed values to replicate the same dev/test split.
    if dataset == 'BBN':
        time = 833365
    else:
        time = 536254
    print("Using seed", time)
    random.seed(time)
    sanitized_mentions['dev'] = set(random.sample(sanitized_mentions['test'], dev_choices))
    sanitized_mentions['test'] = set(sanitized_mentions['test'])
    sanitized_mentions['train'] = set(sanitized_mentions['train'])
    sanitized_mentions['test'] = sanitized_mentions['test'].difference(sanitized_mentions['dev'])
    with open(sanitized_directory + dataset + '/sanitized_mention_dev.txt', 'w') as file_p:
        file_p.write('\n'.join(sorted(list(sanitized_mentions['dev']))))
    with open(sanitized_directory + dataset + '/sanitized_mention_test.txt', 'w') as file_p:
        file_p.write('\n'.join(sorted(list(sanitized_mentions['test']))))
    with open(sanitized_directory + dataset + '/sanitized_mention_train.txt', 'w') as file_p:
        file_p.write('\n'.join(sorted(list(sanitized_mentions['train']))))
    return sanitized_mentions

def generate_label_set(dataset, data_directory, sanitized_directory):
    """
    Generate and save list of unique labels used in a file.
    """
    file_path = data_directory + dataset + '/train_new.json'
    unique_labels = set()
    with open(file_path, 'r') as file_p:
        for row in file_p:
            data = json.loads(row)
            for mention in data['mentions']:
                labels = mention['labels']
                unique_labels.update(labels)
    with open(sanitized_directory + dataset + '/sanitized_labels.txt', 'w') as file_p:
        file_p.write('\n'.join(sorted(list(unique_labels))))

    return unique_labels

def generate_pos_and_dep_set(dataset, data_directory, sanitized_directory):
    """
    Generate and save list of unique pos tag and dep type used in a file.
    """
    file_path = data_directory + dataset + '/train_new.json'
    unique_pos = set()
    unique_dep_type = set()
    with open(file_path, 'r') as file_p:
        for row in file_p:
            data = json.loads(row)
            unique_pos.update(data['pos'])
            for dep in data['dep']:
                unique_dep_type.add(dep['type'])
    with open(sanitized_directory + dataset + '/sanitized_pos.txt', 'w') as file_p:
        file_p.write('\n'.join(sorted(list(unique_pos))))
    with open(sanitized_directory + dataset + '/sanitized_dep_type.txt', 'w') as file_p:
        file_p.write('\n'.join(sorted(list(unique_dep_type))))

def sanitize(file_path, mention_set, output_file_path, label_set):
    """
    Sanitize data.
    """
    file_p_new_json = open(output_file_path, 'w')
    used_mentions = set()
    with open(file_path, 'r') as file_p:
        for row in file_p:
            data = json.loads(row)
            new_mentions = []
            for mention in data['mentions']:
                uid = '_'.join([data['fileid'],
                                str(data['senid']),
                                str(mention['start']),
                                str(mention['end'])
                               ])
                new_labels = set()
                if uid in mention_set and uid not in used_mentions:
                    for label in mention['labels']:
                        if label in label_set:
                            new_labels.add(label)
                    mention['labels'] = list(new_labels)
                    new_mentions.append(dict(mention))
                    used_mentions.add(uid)
            new_row = {}
            new_row['tokens'] = data['tokens']
            new_row['pos'] = data['pos']
            new_row['dep'] = data['dep']
            new_row['mentions'] = new_mentions
            new_row['senid'] = data['senid']
            new_row['fileid'] = data['fileid']
            json.dump(new_row, file_p_new_json)
            file_p_new_json.write('\n')
    file_p_new_json.close()

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: dataset data_directory dev_percentage sanitize_directory')
        sys.exit(0)
    else:
        print('Generating entity mentions.')
        SM = generate_sanitized_mentions(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
        print('Generating label set.')
        UL = generate_label_set(sys.argv[1], sys.argv[2], sys.argv[4])
        print('Generating pos and dep types.')
        generate_pos_and_dep_set(sys.argv[1], sys.argv[2], sys.argv[4])
        print('Sanitizing training data.')
        sanitize(sys.argv[2] + sys.argv[1] + '/train_new.json',
                 SM['train'],
                 sys.argv[4] + sys.argv[1] + '/sanitized_train.json',
                 UL
                )
        print('Sanitizing testing data.')
        sanitize(sys.argv[2] + sys.argv[1] + '/test_new.json',
                 SM['dev'],
                 sys.argv[4] + sys.argv[1] + '/sanitized_dev.json',
                 UL
                )
        print('Sanitizing development data.')
        sanitize(sys.argv[2] + sys.argv[1] + '/test_new.json',
                 SM['test'],
                 sys.argv[4] + sys.argv[1] + '/sanitized_test.json',
                 UL
                )
