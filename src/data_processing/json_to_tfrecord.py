"""
Convert sanitized json data to tfrecord data format.
"""

import sys
import collections
import json
import pickle
import numpy as np
import tensorflow as tf

def invert_dict(dictionary):
    """
    Invert a dict object.
    """
    return {v:k for k, v in dictionary.items()}

def _read_words(filepath):
    """
    Return word list in tokens of json file.
    """
    words = []
    with open(filepath, 'r', encoding='utf-8') as file_p:
        for row in file_p:
            words.extend(json.loads(row)['tokens'])
    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], str(x[0])))
    words, counts = list(zip(*count_pairs))
    return words, counts

def _read_mention_chars(filepath, remove_below):
    """
    Return character list in mentions of json file.
    """
    char_list = []
    with open(filepath, 'r', encoding='utf-8') as file_p:
        for row in file_p:
            json_data = json.loads(row)
            tokens = json_data['tokens']
            for mention in json_data['mentions']:
                for char in ' '.join(tokens[mention['start']:mention['end']]):
                    char_list.append(char)

    counter = collections.Counter(char_list)

    chrs, counts = list(zip(*sorted(counter.items(), key=lambda x: (-x[1], x[0]))))

    chrs = np.array(chrs)
    counts = np.array(counts)

    # remove infrequent characters
    mask = counts >= remove_below
    chrs = chrs[mask]
    counts = counts[mask]

    # 0th character will be used as padding
    num_to_chrs = dict(enumerate(chrs, 1))

    # add unique character
    num_to_chrs[len(num_to_chrs) + 1] = 'unk'
    # add end of mention character
    num_to_chrs[len(num_to_chrs) + 1] = 'eos'

    chrs_to_num = invert_dict(num_to_chrs)

    return chrs_to_num

def load_filtered_embeddings(filepath, word_list):
    """
    Load selected pre-trained word vectors based on word list.
    """
    word_dic = {}
    word_found = set()
    word_set = set(word_list)
    with open(filepath, 'r', encoding='utf-8') as file_p:
        for line in file_p:
            splits = line.split(' ')
            word = splits[0]
            if word in word_set or word == 'unk':
                word_dic[word] = [float(x) for x in splits[1:]]
                word_found.add(word)
    word_not_found = word_set.difference(word_found)
    # enumeration will start from 1
    word_to_num = dict(zip(word_dic.keys(), range(1, len(word_dic) + 1)))

    # 0th pre_trained_embedding will be remain 0
    pre_trained_embeddings = np.zeros((len(word_to_num) + 1, len(word_dic['unk'])),
                                      dtype=np.core.numerictypes.float32
                                     )
    for word in word_to_num:
        pre_trained_embeddings[word_to_num[word]] = word_dic[word]
    return word_to_num, pre_trained_embeddings, word_not_found

def generate_labels_to_numbers(dataset, sanitized_directory):
    """
    Generate label to number dictionary.
    """
    with open(sanitized_directory + dataset + '/sanitized_labels.txt', 'r') as file_p:
        label_list = file_p.read().split('\n')
        num_to_label = dict(zip(label_list, range(len(label_list))))
        return num_to_label

def generate_features_to_numbers(dataset, sanitized_directory):
    """
    Generate pos and dep type to number dictionary.
    """
    with open(sanitized_directory + dataset + '/sanitized_pos.txt', 'r') as file_p:
        pos_list = file_p.read().split('\n')
        num_to_pos = dict(zip(pos_list, range(len(pos_list))))
    with open(sanitized_directory + dataset + '/sanitized_dep_type.txt', 'r') as file_p:
        dep_type_list = file_p.read().split('\n')
        num_to_dep_type = dict(zip(dep_type_list, range(len(dep_type_list))))

    return num_to_pos, num_to_dep_type

def labels_status(labels):
    """
    Check is labels is clean or not.
    """
    leaf = max(labels, key=lambda x: x.count('/'))
    clean = 1
    for label in labels:
        if label not in leaf:
            clean = 0
    return clean


#pylint: disable-msg=R0914
def make_tf_record_f1(json_data, mention, mappings):
    """
    A tfrecord per mention.
    """
    start = mention['start']
    end = mention['end']

    tokens = json_data['tokens']
    poss = json_data['pos']
    dep_types = json_data['dep']

    uid = bytes('_'.join([json_data['fileid'],
                          str(json_data['senid']),
                          str(start),
                          str(end)
                         ]), 'utf-8')
    # lc and rc include mention

    left_context = tokens[:end]
    entity = tokens[start:end]
    right_context = tokens[start:]

    left_poss = poss[:end]
    right_poss = poss[start:]

    left_dts = dep_types[:end]
    right_dts = dep_types[start:]

    ex = tf.train.SequenceExample()

    ex.context.feature["uid"].bytes_list.value.append(uid)
    ex.context.feature["lcl"].int64_list.value.append(len(left_context))
    ex.context.feature["rcl"].int64_list.value.append(len(right_context))
    ex.context.feature["eml"].int64_list.value.append(len(' '.join(entity)) + 1)
    ex.context.feature["clean"].int64_list.value.append(labels_status(mention['labels']))

    lc_ids = ex.feature_lists.feature_list["lci"]
    rc_ids = ex.feature_lists.feature_list["rci"]
    em_ids = ex.feature_lists.feature_list["emi"]
    l_pos_ids = ex.feature_lists.feature_list["lpi"]
    r_pos_ids = ex.feature_lists.feature_list["rpi"]
    l_dt_ids = ex.feature_lists.feature_list["ldti"]
    r_dt_ids = ex.feature_lists.feature_list["rdti"]
    label_list = ex.feature_lists.feature_list["labels"]

    for word in left_context:
        lc_ids.feature.add().int64_list.value.append(mappings['wtn'].get(word,
                                                                         mappings['wtn']['unk']))
    for word in right_context:
        rc_ids.feature.add().int64_list.value.append(mappings['wtn'].get(word,
                                                                         mappings['wtn']['unk']))
    for char in ' '.join(entity):
        em_ids.feature.add().int64_list.value.append(mappings['ctn'].get(char,
                                                                         mappings['ctn']['unk']))
    em_ids.feature.add().int64_list.value.append(mappings['ctn']['eos'])

    for pos in left_poss:
        l_pos_ids.feature.add().int64_list.value.append(mappings['ptn'][pos])

    for pos in right_poss:
        r_pos_ids.feature.add().int64_list.value.append(mappings['ptn'][pos])

    for dep_type in left_dts:
        l_dt_ids.feature.add().int64_list.value.append(mappings['dttn'][dep_type['type']])

    for dep_type in right_dts:
        # small hack, get(dep_type, 0) need to fix this when doing transfer learning
        # with Wiki and OntoNotes dataset
        # conj:uh not found in Wiki dataset
        # For all other experiments, this will not affect
        r_dt_ids.feature.add().int64_list.value.append(mappings['dttn'].get(dep_type['type'], 0))

    temp_labels = [0] * len(mappings['ltn'])
    for label in mention['labels']:
        temp_labels[mappings['ltn'][label]] = 1
    for label in temp_labels:
        label_list.feature.add().int64_list.value.append(label)
    return ex

#pylint: disable-msg=R0914
def make_tf_record_f2(json_data, mention, mappings, mention_window, context_window):
    """
    A tfrecord per mention.
    """
    start = mention['start']
    end = mention['end']

    tokens = json_data['tokens']

    uid = bytes('_'.join([json_data['fileid'],
                          str(json_data['senid']),
                          str(start),
                          str(end)
                         ]), 'utf-8')

    # lc and rc does not include mention
    # as mentioned in AKBC paper
    if context_window:
        left_context = tokens[:start][-context_window:]
        right_context = tokens[end:][:context_window]
    else:
        left_context = tokens[:start]
        right_context = tokens[end:]
    if mention_window:
        entity = tokens[start:end][:mention_window]
    else:
        entity = tokens[start:end]

    ex = tf.train.SequenceExample()

    ex.context.feature["uid"].bytes_list.value.append(uid)
    ex.context.feature["lcl"].int64_list.value.append(len(left_context))
    ex.context.feature["rcl"].int64_list.value.append(len(right_context))
    ex.context.feature["eml"].int64_list.value.append(len(entity))
    # This will only be used in representations experiment.
    ex.context.feature["clean"].int64_list.value.append(labels_status(mention['labels']))

    lc_ids = ex.feature_lists.feature_list["lci"]
    rc_ids = ex.feature_lists.feature_list["rci"]
    em_ids = ex.feature_lists.feature_list["emi"]
    label_list = ex.feature_lists.feature_list["labels"]

    for word in left_context:
        lc_ids.feature.add().int64_list.value.append(mappings['wtn'].get(word,
                                                                         mappings['wtn']['unk']))
    for word in right_context:
        rc_ids.feature.add().int64_list.value.append(mappings['wtn'].get(word,
                                                                         mappings['wtn']['unk']))
    for word in entity:
        em_ids.feature.add().int64_list.value.append(mappings['wtn'].get(word,
                                                                         mappings['wtn']['unk']))

    temp_labels = [0] * len(mappings['ltn'])
    for label in mention['labels']:
        temp_labels[mappings['ltn'][label]] = 1
    for label in temp_labels:
        label_list.feature.add().int64_list.value.append(label)
    return ex

def data_format_f1(in_filepath, out_filepath, mappings):
    """
    Convert json file to tfrecord.
    """
    total = 0
    with open(in_filepath, 'r') as file_p1, open(out_filepath, 'wb') as file_p2:
        writer = tf.python_io.TFRecordWriter(file_p2.name)
        for row in file_p1:
            json_data = json.loads(row)
            for mention in json_data['mentions']:
                ex = make_tf_record_f1(json_data, mention, mappings)
                writer.write(ex.SerializeToString())
                total += 1
        writer.close()
    return total

def data_format_f2(in_filepath, out_filepath, mappings):
    """
    Convert json file to tfrecord.
    """
    total = 0
    with open(in_filepath, 'r') as file_p1, open(out_filepath, 'wb') as file_p2:
        writer = tf.python_io.TFRecordWriter(file_p2.name)
        for row in file_p1:
            json_data = json.loads(row)
            for mention in json_data['mentions']:
                # window width as mentioned in AKBC paper
                ex = make_tf_record_f2(json_data, mention, mappings, 5, 15)
                writer.write(ex.SerializeToString())
                total += 1
        writer.close()
    return total

def data_format_f5(in_filepath, out_filepath, mappings):
    """
    Convert json file to tfrecord.
    """
    total = 0
    with open(in_filepath, 'r') as file_p1, open(out_filepath, 'wb') as file_p2:
        writer = tf.python_io.TFRecordWriter(file_p2.name)
        for row in file_p1:
            json_data = json.loads(row)
            for mention in json_data['mentions']:
                ex = make_tf_record_f2(json_data, mention, mappings, None, None)
                writer.write(ex.SerializeToString())
                total += 1
        writer.close()
    return total

def data_format_abhishek(dataset, sanitized_directory, glove_vector_filepath, output_directory):
    """
    Generate data as needed by our model.
    """
    print('Reading words.')
    words, _ = _read_words(sanitized_directory + dataset + '/sanitized_train.json')
    print('Loading word embeddings.')
    word_to_num, embedding, _ = load_filtered_embeddings(glove_vector_filepath, words)
    print('Embedding shape', embedding.shape)
    print('Generating label to number dictionary.')
    label_to_num = generate_labels_to_numbers(dataset, sanitized_directory)
    print('Generating pos and dep type to number dictionary.')
    pos_to_num, dep_type_to_num = generate_features_to_numbers(dataset, sanitized_directory)
    print('Generating character to number dictionary.')
    chrs_to_num = _read_mention_chars(sanitized_directory + dataset + '/sanitized_train.json', 5)

    mappings = {}
    mappings['wtn'] = word_to_num
    mappings['ctn'] = chrs_to_num
    mappings['ltn'] = label_to_num
    mappings['ptn'] = pos_to_num
    mappings['dttn'] = dep_type_to_num
    print('Generating training data.')
    train_size = data_format_f1(sanitized_directory + dataset + '/sanitized_train.json',
                                output_directory + 'f1/' + dataset + '/train.tfrecord',
                                mappings
                               )
    print('Generating development data.')
    dev_size = data_format_f1(sanitized_directory + dataset + '/sanitized_dev.json',
                              output_directory + 'f1/' + dataset + '/dev.tfrecord',
                              mappings
                             )
    print('Generating testing data.')
    test_size = data_format_f1(sanitized_directory + dataset + '/sanitized_test.json',
                               output_directory + 'f1/' + dataset + '/test.tfrecord',
                               mappings
                              )

    pickle.dump({
        'num_to_label': invert_dict(label_to_num),
        'num_to_word' : invert_dict(word_to_num),
        'num_to_chrs' : invert_dict(chrs_to_num),
        'num_to_pos' : invert_dict(pos_to_num),
        'num_to_dep_type' : invert_dict(dep_type_to_num),
        'word_embedding' : embedding,
        'train_size' : train_size,
        'dev_size' : dev_size,
        'test_size' : test_size
    }, open(output_directory + 'f1/' + dataset + '/local_variables.pickle', 'wb'))

def data_format_shimaoka(dataset, sanitized_directory, glove_vector_filepath, output_directory):
    """
    Generate data as needed by shimaoka model.
    """
    print('Reading words.')
    words, _ = _read_words(sanitized_directory + dataset + '/sanitized_train.json')
    print('Loading word embeddings.')
    word_to_num, embedding, _ = load_filtered_embeddings(glove_vector_filepath, words)
    print('Embedding shape', embedding.shape)
    print('Generating label to number dictionary.')
    label_to_num = generate_labels_to_numbers(dataset, sanitized_directory)

    mappings = {}
    mappings['wtn'] = word_to_num
    mappings['ltn'] = label_to_num
    print('Generating training data.')
    train_size = data_format_f2(sanitized_directory + dataset + '/sanitized_train.json',
                                output_directory + 'f2/' + dataset + '/train.tfrecord',
                                mappings
                               )
    print('Generating development data.')
    dev_size = data_format_f2(sanitized_directory + dataset + '/sanitized_dev.json',
                              output_directory + 'f2/' + dataset + '/dev.tfrecord',
                              mappings
                             )
    print('Generating testing data.')
    test_size = data_format_f2(sanitized_directory + dataset + '/sanitized_test.json',
                               output_directory + 'f2/' + dataset + '/test.tfrecord',
                               mappings
                              )

    pickle.dump({
        'num_to_label': invert_dict(label_to_num),
        'num_to_word' : invert_dict(word_to_num),
        'word_embedding' : embedding,
        'train_size' : train_size,
        'dev_size' : dev_size,
        'test_size' : test_size
    }, open(output_directory + 'f2/' + dataset + '/local_variables.pickle', 'wb'))
#pylint: disable=invalid-name
def data_format_shimaoka_representation(dataset,
                                        sanitized_directory,
                                        glove_vector_filepath,
                                        output_directory):
    """
    Generate data as needed by shimaoka model.
    """
    print('Reading words.')
    words, _ = _read_words(sanitized_directory + dataset + '/sanitized_train.json')
    print('Loading word embeddings.')
    word_to_num, embedding, _ = load_filtered_embeddings(glove_vector_filepath, words)
    print('Embedding shape', embedding.shape)
    print('Generating label to number dictionary.')
    label_to_num = generate_labels_to_numbers(dataset, sanitized_directory)

    mappings = {}
    mappings['wtn'] = word_to_num
    mappings['ltn'] = label_to_num
    print('Generating training data.')
    train_size = data_format_f5(sanitized_directory + dataset + '/sanitized_train.json',
                                output_directory + 'f5/' + dataset + '/train.tfrecord',
                                mappings
                               )
    print('Generating development data.')
    dev_size = data_format_f5(sanitized_directory + dataset + '/sanitized_dev.json',
                              output_directory + 'f5/' + dataset + '/dev.tfrecord',
                              mappings
                             )
    print('Generating testing data.')
    test_size = data_format_f5(sanitized_directory + dataset + '/sanitized_test.json',
                               output_directory + 'f5/' + dataset + '/test.tfrecord',
                               mappings
                              )

    pickle.dump({
        'num_to_label': invert_dict(label_to_num),
        'num_to_word' : invert_dict(word_to_num),
        'word_embedding' : embedding,
        'train_size' : train_size,
        'dev_size' : dev_size,
        'test_size' : test_size
    }, open(output_directory + 'f5/' + dataset + '/local_variables.pickle', 'wb'))

def data_format_transfer_learning(dataset, sanitized_directory, output_directory):
    """
    Generate data as needed for finetuning.
    """
    # Wiki dataset hard coded.
    l_vars = pickle.load(open(output_directory + 'f1/Wiki/' +  'local_variables.pickle', 'rb'))
    embedding = l_vars['word_embedding']
    print('Embedding shape', embedding.shape)

    print('Generating label to number dictionary.')
    label_to_num = generate_labels_to_numbers(dataset, sanitized_directory)

    word_to_num = invert_dict(l_vars['num_to_word'])
    chrs_to_num = invert_dict(l_vars['num_to_chrs'])
    pos_to_num = invert_dict(l_vars['num_to_pos'])
    dep_type_to_num = invert_dict(l_vars['num_to_dep_type'])

    mappings = {}
    mappings['wtn'] = word_to_num
    mappings['ctn'] = chrs_to_num
    mappings['ltn'] = label_to_num
    mappings['ptn'] = pos_to_num
    mappings['dttn'] = dep_type_to_num
    print('Generating training data.')
    train_size = data_format_f1(sanitized_directory + dataset + '/sanitized_train.json',
                                output_directory + 'f3/' + dataset + '/train.tfrecord',
                                mappings
                               )
    print('Generating development data.')
    dev_size = data_format_f1(sanitized_directory + dataset + '/sanitized_dev.json',
                              output_directory + 'f3/' + dataset + '/dev.tfrecord',
                              mappings
                             )
    print('Generating testing data.')
    test_size = data_format_f1(sanitized_directory + dataset + '/sanitized_test.json',
                               output_directory + 'f3/' + dataset + '/test.tfrecord',
                               mappings
                              )

    pickle.dump({
        'num_to_label': invert_dict(label_to_num),
        'num_to_word' : invert_dict(word_to_num),
        'num_to_chrs' : invert_dict(chrs_to_num),
        'num_to_pos' : invert_dict(pos_to_num),
        'num_to_dep_type' : invert_dict(dep_type_to_num),
        'word_embedding' : embedding,
        'train_size' : train_size,
        'dev_size' : dev_size,
        'test_size' : test_size
    }, open(output_directory + 'f3/' + dataset + '/local_variables.pickle', 'wb'))

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Usage: dataset sanitized_directory glove_vector_filepath format output_directory')
        sys.exit(0)
    else:
        FORMAT = sys.argv[4]
        if FORMAT == 'f1':
            data_format_abhishek(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[5])
        elif FORMAT == 'f2':
            data_format_shimaoka(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[5])
        elif FORMAT == 'f3':
            data_format_transfer_learning(sys.argv[1], sys.argv[2], sys.argv[5])
        elif FORMAT == 'f5':
            data_format_shimaoka_representation(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[5])
