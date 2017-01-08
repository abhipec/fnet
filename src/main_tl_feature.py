"""
Main code to run our experiments.
Usage:
    main_our [options]
    main_our -h | --help

Options:
    -h, --help                      Print this.
    --dataset=<d>                   Dataset.
    --data_directory=<dir>          Data directory.
    --learning_rate=<value>
    --joint_embedding_size=<size>
    --epochs=<number>
    --threshold=<number>
    --batch_size=<number>
    --uid=<unique_id>
    --use_clean

"""
import os
import sys
import json
from docopt import docopt
import numpy as np
import tensorflow as tf
#pylint: disable=no-member
import pandas as pd
#pylint: disable=import-error
import models.AFET as model
from models.evalution import hierarchical_prediction
from models.metrics import strict, loose_macro, loose_micro
import plotly
#pylint: disable=no-member
import plotly.graph_objs as go

#pylint: disable=too-many-arguments, too-many-locals
def run_one_epoch(batch,
                  inputs,
                  operations,
                  batch_size,
                  model_parameters,
                  session,
                  is_training=False):
    """
    Run a single batch over dataset.
    """
    features = batch['features']
    targets = batch['targets']
    clean = batch['clean']
    mentions = batch['mentions']

    data_size = features.shape[0]

    # random shuffle batch
    per = np.random.permutation(data_size)
    features = features[per]
    targets = targets[per]
    clean = clean[per]
    mentions = mentions[per]

    num_of_batches = ((data_size - 1) // batch_size) + 1
    # need to store predictions of not doing training
    if not is_training:
        predictions = np.empty((0, model_parameters.output_dim),
                               dtype=np.core.numerictypes.float32)

    total_cost = []
    for i in range(num_of_batches):
        features_b = features[i * batch_size:(i + 1) * batch_size]
        targets_b = targets[i * batch_size:(i + 1) * batch_size]
        clean_b = clean[i * batch_size:(i + 1) * batch_size]

        feed_dict = {
            inputs['features']: features_b,
            inputs['labels']: targets_b,
            inputs['clean']: clean_b
        }
        if is_training:
            cost, _ = session.run([operations['cost'], operations['optimize']], feed_dict=feed_dict)
        else:
            cost, scores = session.run([operations['cost'], operations['prediction']],
                                       feed_dict=feed_dict)
            predictions = np.vstack([predictions, scores])
        total_cost.append(cost)
        sys.stdout.write('\r{} / {} : mean epoch cost = {}'.format(i,
                                                                   num_of_batches,
                                                                   np.mean(total_cost)))
        sys.stdout.flush()
        sys.stdout.write('\r')
    if is_training:
        return {
            'cost': np.mean(total_cost)
        }
    else:
        return {
            'cost': np.mean(total_cost),
            'predictions' : predictions,
            'targets' : targets,
            'mentions' : mentions
        }

def plot_dataframe(data_frame, filename):
    """
    Plot the results in file.
    """
    x_axis = list(range(len(data_frame)))
    columns = list(data_frame.columns.values)

    traces = []
    for column in columns:
        traces.append(go.Scatter(x=x_axis,
                                 y=data_frame[column],
                                 mode='lines',
                                 name=column))

    fig = go.Figure(data=traces)
    plotly.offline.plot(fig, filename=filename, auto_open=False)

def log_results(results, num_to_label, log_directory, epoch, threshold):
    """
    Log the results in a file.
    """
    log_file = log_directory + '/data_log.csv'
    result_file = log_directory + '/result_' + str(epoch) + '.txt'
    image_file = log_directory + '/image'

    if os.path.isfile(log_file):
        data = pd.read_csv(log_file)
    else:
        data = pd.DataFrame(columns=('train_cost',
                                     'dev_cost',
                                     'test_cost',
                                     'dev_acc',
                                     'dev_ma_F1',
                                     'dev_mi_F1',
                                     'test_acc',
                                     'test_ma_F1',
                                     'test_mi_F1'))

    current_result = []
    current_result.append(results['train']['cost'])
    current_result.append(results['dev']['cost'])
    current_result.append(results['test']['cost'])

    new_predictions = hierarchical_prediction(results['dev']['predictions'], num_to_label, threshold)

    current_result.extend(compute_metrics(new_predictions,
                                          results['dev']['targets']))

    new_predictions = hierarchical_prediction(results['test']['predictions'], num_to_label, threshold)

    current_result.extend(compute_metrics(new_predictions,
                                          results['test']['targets']))
    data.loc[len(data)] = current_result
    data.to_csv(log_file, index=False)
    plot_dataframe(data, image_file)
    with open(result_file, 'w') as file_p:
        for i in range(len(new_predictions)):
            labels = [num_to_label[x] for x in np.where(new_predictions[i] > 0)[0]]
            file_p.write(str(results['test']['mentions'][i], 'utf-8')
                         + '\t' + ','.join(labels) + '\n')

def compute_metrics(predictions, targets):
    """
    Compute metrics as required.
    """
    return (strict(predictions, targets),
            loose_macro(predictions, targets),
            loose_micro(predictions, targets))

def one_train_dev_test_epoch(representations,
                             inputs,
                             operations,
                             model_parameters,
                             session):
    """
    Run one iteration of training epoch.
    """
    batch_size = representations['batch_size']

    # training epoch
    print('Training')
    results = {}
    results['train'] = run_one_epoch(representations['train'],
                                     inputs,
                                     operations,
                                     batch_size,
                                     model_parameters['train'],
                                     session,
                                     is_training=True)

    print('Development')
    results['dev'] = run_one_epoch(representations['dev'],
                                   inputs,
                                   operations,
                                   batch_size,
                                   model_parameters['dev'],
                                   session,
                                   is_training=False)
    print('Test')
    results['test'] = run_one_epoch(representations['test'],
                                    inputs,
                                    operations,
                                    batch_size,
                                    model_parameters['test'],
                                    session,
                                    is_training=False)

    return results

#pylint: disable=invalid-name
if __name__ == '__main__':
    tf.reset_default_graph()
    arguments = docopt(__doc__)
    print(arguments)

    bs = int(arguments['--batch_size'])
    relevant_directory = os.path.expanduser(
        arguments['--data_directory']) + arguments['--dataset'] + '/'
    all_representations, parameters = model.read_local_variables_and_params(arguments)
    all_representations['batch_size'] = bs

    placeholders = model.create_placeholders(parameters['train'])
    cmatrix = np.ones((parameters['train'].output_dim, parameters['train'].output_dim),
                      dtype=np.core.numerictypes.float32)
    ops = model.model(placeholders,
                      parameters['train'],
                      cmatrix,
                      is_training=True)

    sess = tf.Session()
    # Create a coordinator, launch the queue runner threads.
    coord = tf.train.Coordinator()
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    epochs_elapsed = 0
    ckpt_directory = os.path.join(os.path.dirname(__file__), '../', 'ckpt', arguments['--uid'])
    os.makedirs(ckpt_directory, exist_ok=True)
    # dump parameters used to disk
    with open(ckpt_directory + '/parameters.json', 'w') as json_p:
        json.dump(arguments, json_p, sort_keys=True, indent=4)

    summary_writer = tf.train.SummaryWriter(ckpt_directory + '/graph/', sess.graph)

    try:
        while not coord.should_stop():
            # Run training steps
            if epochs_elapsed >= int(arguments['--epochs']):
                print('MAX epoch completed using ckpt model.')
                coord.request_stop()
                break
            result = one_train_dev_test_epoch(all_representations,
                                              placeholders,
                                              ops,
                                              parameters,
                                              sess)
            epochs_elapsed += 1
            log_results(result,
                        all_representations['num_to_label'],
                        ckpt_directory,
                        epochs_elapsed,
                        float(arguments['--threshold']))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    except tf.errors.CancelledError:
        print('Done training -- epoch limit reached counting checkpoints')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    sess.close()
