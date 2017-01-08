"""
Visualize label distribution for training, test and developemnt dataset.abs
Usage:
    parse_record <dataset>
    parse_record -h | --help

Options:
    -h, --help                      Print this.

"""
import pickle
from docopt import docopt
import numpy as np
import tensorflow as tf
import plotly
#pylint: disable=no-member
import plotly.graph_objs as go

def class_wise_clean_percentage(data, clean, num_to_label, filename):
    """
    Class wise analysis of clean and not clean distribution.
    """
    data_list = []
    for label_no in num_to_label:
        ids = data[:, label_no] == 1
        filtered_clean = clean[ids]
        data_list.append((num_to_label[label_no],
                          (np.sum(filtered_clean) * 100)/ len(filtered_clean)))
    sorted_list = sorted(data_list, key=lambda x: x[1])
    with open(filename, 'w') as file_p:
        for row in sorted_list:
            file_p.write(row[0] + '\t')
            file_p.write(str(row[1]) + '\n')
    return list(sorted_list)

def labels_below_x_percentage(labels, sorted_list, num_to_label, below_percentage):
    """
    Print labels that are below x percentage.
    """
    for label_no in num_to_label:
        if np.sum(labels[:, label_no]):
            for percentage in sorted_list:
                if percentage[0] == num_to_label[label_no] and percentage[1] < below_percentage:
                    print(num_to_label[label_no])

#pylint: disable=too-many-locals
def return_labels(tfrecord_filename, local_variables_filename):
    """
    Return labels in filename specified.
    """
    context_features = {
        "lcl": tf.FixedLenFeature([], dtype=tf.int64),
        "rcl": tf.FixedLenFeature([], dtype=tf.int64),
        "eml": tf.FixedLenFeature([], dtype=tf.int64),
        "uid": tf.FixedLenFeature([], dtype=tf.string),
        "clean": tf.FixedLenFeature([], dtype=tf.int64)}
    sequence_features = {
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "lci": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "rci": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "emi": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "lpi": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "rpi": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "ldti": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "rdti": tf.FixedLenSequenceFeature([], dtype=tf.int64)}

    tf.reset_default_graph()
    l_vars = pickle.load(open(local_variables_filename, 'rb'))
    label_dim = len(l_vars['num_to_label'])
    num_to_label = l_vars['num_to_label']

    filename_queue = tf.train.string_input_producer(
        [tfrecord_filename],
        num_epochs=1)
    reader = tf.TFRecordReader()

    _, ex = reader.read(filename_queue)

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=ex,
        context_features=context_features,
        sequence_features=sequence_features
        )

    all_examples = {}
    all_examples.update(context_parsed)
    all_examples.update(sequence_parsed)

    batched_data = tf.train.batch(
        tensors=all_examples,
        batch_size=50000,
        dynamic_pad=True,
        allow_smaller_final_batch=True,
        name="y_batch"
    )
    sess = tf.Session()

    # Create a coordinator, launch the queue runner threads.
    coord = tf.train.Coordinator()
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    tf.train.start_queue_runners(sess=sess)
    labels = np.empty((0, label_dim))
    clean = np.empty((0))
    try:
        while not coord.should_stop():
            # Run training steps or whatever
            out = sess.run(batched_data)
            labels = np.vstack((labels, out['labels']))
            clean = np.hstack((clean, out['clean']))
            print(labels.shape)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    # And wait for them to actually do it.
    sess.close()
    return labels, num_to_label, clean
#pylint: disable-msg=C0103
if __name__ == '__main__':
    dataset = docopt(__doc__)['<dataset>']

    relevant_directory = '../data/processed/f1/' + dataset + '/'
    train_labels, ntl, train_clean = return_labels(relevant_directory + 'train.tfrecord',
                                                   relevant_directory + 'local_variables.pickle')
    test_labels, _, test_clean = return_labels(relevant_directory + 'test.tfrecord',
                                               relevant_directory + 'local_variables.pickle')
    dev_labels, _, dev_clean = return_labels(relevant_directory + 'dev.tfrecord',
                                             relevant_directory + 'local_variables.pickle')

    print('Percentage of clean labels, train set, ', (100 * np.sum(train_clean)) / len(train_clean))
    print('Percentage of clean labels, test set, ', (100 * np.sum(test_clean)) / len(test_clean))
    print('Percentage of clean labels, dev set, ', (100 * np.sum(dev_clean)) / len(dev_clean))

    class_wise_clean_percentage(train_labels,
                                train_clean,
                                ntl,
                                '../stats/class_wise_clean_percentage_' + dataset + '.txt')

    x_axis = [ntl[i] for i in range(len(ntl))]

    train_sum = np.sum(train_labels)
    y_train = [np.round(np.sum(train_labels[:, i]) * 100 / train_sum, 2) for i in range(len(ntl))]
    trace0 = go.Scatter(x=x_axis,
                        y=y_train,
                        mode='lines',
                        name='train'
                       )

    test_sum = np.sum(test_labels)
    y_test = [np.round(np.sum(test_labels[:, i]) * 100 / test_sum, 2) for i in range(len(ntl))]
    trace1 = go.Scatter(x=x_axis,
                        y=y_test,
                        mode='lines',
                        name='test'
                       )

    dev_sum = np.sum(dev_labels)
    y_dev = [np.round(np.sum(dev_labels[:, i]) * 100 / dev_sum, 2) for i in range(len(ntl))]
    trace2 = go.Scatter(x=x_axis,
                        y=y_dev,
                        mode='lines',
                        name='development'
                       )

    fig = go.Figure(data=[trace0, trace1, trace2])
    plotly.offline.plot(fig, filename='../stats/' + dataset + '_label_plot', auto_open=False)
