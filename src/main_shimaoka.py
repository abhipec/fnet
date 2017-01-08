"""
Main code to run our experiments.
Usage:
    main_our [options]
    main_our -h | --help

Options:
    -h, --help                      Print this.
    --dataset=<d>                   Dataset.
    --data_directory=<dir>          Data directory.
    --rnn_hidden_neurons=<size>
    --keep_prob=<value>
    --learning_rate=<value>
    --epochs=<number>
    --batch_size=<number>
    --attention_size=<number>
    --uid=<unique_id>
    --retrain_word_embeddings

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
import models.ShimaokaClassificationModel as model
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
    batches_elapsed = 0
    total_batches = model_parameters.data_size // batch_size + 1
    # need to store predictions of not doing training
    if not is_training:
        predictions = np.empty((0, model_parameters.output_dim),
                               dtype=np.core.numerictypes.float32)
        targets = np.empty((0, model_parameters.output_dim),
                           dtype=np.core.numerictypes.float32)
        mentions = np.empty((0), dtype=np.object)

    total_cost = []
    while batches_elapsed * batch_size < model_parameters.data_size:
        out = session.run(batch)
        feed_dict = {}
        to_feed = list(inputs.keys())
        feed_dict[inputs['keep_prob']] = model_parameters.keep_prob
        to_feed.remove('keep_prob')

        for key in to_feed:
            feed_dict[inputs[key]] = out[key]
        if is_training:
            cost, _ = session.run([operations['cost'], operations['optimize']], feed_dict=feed_dict)
        else:
            cost, scores = session.run([operations['cost'], operations['prediction']],
                                       feed_dict=feed_dict)
            predictions = np.vstack([predictions, scores])
            targets = np.vstack([targets, out['labels']])
            mentions = np.hstack((mentions, out['uid']))
        batches_elapsed += 1
        total_cost.append(cost)
        sys.stdout.write('\r{} / {} : mean epoch cost = {}'.format(batches_elapsed,
                                                                   total_batches,
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
            'predictions' : predictions[:model_parameters.data_size,],
            'targets' : targets[:model_parameters.data_size,],
            'mentions' : mentions[:model_parameters.data_size,]
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

def log_results(results, num_to_label, log_directory, epoch):
    """
    Log the results in a file.
    """
    log_file = log_directory + '/data_log.csv'
    result_file = log_directory + '/result_' + str(epoch) + '.txt'
    result_file_dev = log_directory + '/result_dev_' + str(epoch) + '.txt'
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

    new_predictions = results['dev']['predictions']
    # Make sure at-least one output
    c_argmax = np.argmax(new_predictions, 1)[:, np.newaxis]
    b_map = c_argmax == np.arange(new_predictions.shape[1])
    new_predictions[b_map] = 1
    new_predictions[new_predictions > 0.5] = 1
    new_predictions[new_predictions <= 0.5] = 0

    current_result.extend(compute_metrics(new_predictions,
                                          results['dev']['targets']))

    with open(result_file_dev, 'w') as file_p:
        for i in range(len(new_predictions)):
            labels = [num_to_label[x] for x in np.where(new_predictions[i] > 0)[0]]
            file_p.write(str(results['dev']['mentions'][i], 'utf-8')
                         + '\t' + ','.join(labels) + '\n')

    new_predictions = results['test']['predictions']
    # Make sure at-least one output
    c_argmax = np.argmax(new_predictions, 1)[:, np.newaxis]
    b_map = c_argmax == np.arange(new_predictions.shape[1])
    new_predictions[b_map] = 1
    new_predictions[new_predictions > 0.5] = 1
    new_predictions[new_predictions <= 0.5] = 0

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

def one_train_dev_test_epoch(batches,
                             inputs,
                             operations,
                             model_parameters,
                             session):
    """
    Run one iteration of training epoch.
    """
    batch_size = batches['size']

    # training epoch
    print('Training')
    results = {}
    results['train'] = run_one_epoch(batches['train'],
                                     inputs,
                                     operations,
                                     batch_size,
                                     model_parameters['train'],
                                     session,
                                     is_training=True)

    print('Development')
    results['dev'] = run_one_epoch(batches['dev'],
                                   inputs,
                                   operations,
                                   batch_size,
                                   model_parameters['dev'],
                                   session,
                                   is_training=False)
    print('Test')
    results['test'] = run_one_epoch(batches['test'],
                                    inputs,
                                    operations,
                                    batch_size,
                                    model_parameters['test'],
                                    session,
                                    is_training=False)

    return results

def load_checkpoint(checkpoint_directory,
                    session,
                    finetune=None,
                    finetune_directory=None):
    """
    Load checkpoint if exists.
    """
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # filter variables if needed.
    saver_ob = tf.train.Saver(variables, max_to_keep=0)
    os.makedirs(checkpoint_directory, exist_ok=True)
    # verify if we don't have a checkpoint saved directly
    step = 0
    ckpt = tf.train.get_checkpoint_state(checkpoint_directory)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        model_checkpoint_path = ckpt.model_checkpoint_path
        saver_ob.restore(session, model_checkpoint_path)
        step = int(model_checkpoint_path.rsplit('-', 1)[1])
        print('Model loaded = ', step)
    elif finetune:
        # if finetune flag is set and no checkpoint available
        # load finetune model from finetune_directory
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        relevant_variables = [v for v in variables if v.name != 'embeddings/label_embeddings:0']
        new_saver = tf.train.Saver(relevant_variables, max_to_keep=0)
        ckpt = tf.train.get_checkpoint_state(finetune_directory)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            model_checkpoint_path = ckpt.model_checkpoint_path
            new_saver.restore(session, model_checkpoint_path)
            step = int(model_checkpoint_path.rsplit('-', 1)[1])
            print('Finetune Model loaded = ', step)

    return saver_ob, step

#pylint: disable=invalid-name
if __name__ == '__main__':
    tf.reset_default_graph()
    arguments = docopt(__doc__)
    print(arguments)
    bs = int(arguments['--batch_size'])
    relevant_directory = os.path.expanduser(
        arguments['--data_directory']) + arguments['--dataset'] + '/'
    l_variables, parameters = model.read_local_variables_and_params(arguments)
    placeholders = model.create_placeholders(parameters['train'])
    ops = model.model(placeholders,
                      parameters['train'],
                      l_variables['word_embedding'],
                      is_training=True)

    data_batches, queue_runners = model.read_batches(relevant_directory, bs)
    sess = tf.Session()
    # Create a coordinator, launch the queue runner threads.
    coord = tf.train.Coordinator()
    # start queue runners
    enqueue_threads = []
    for qr in queue_runners:
        enqueue_threads.append(queue_runners[qr].create_threads(sess, coord=coord, start=True))
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    tf.train.start_queue_runners(sess=sess)

    # Create a saver and session object.
    ckpt_directory = os.path.join(os.path.dirname(__file__), '../', 'ckpt', arguments['--uid'])
    saver, initial_step = load_checkpoint(ckpt_directory,
                                          sess)
    epochs_elapsed = initial_step

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
            result = one_train_dev_test_epoch(data_batches, placeholders, ops, parameters, sess)
            epochs_elapsed += 1
            log_results(result, l_variables['num_to_label'], ckpt_directory, epochs_elapsed)
            saver.save(sess, ckpt_directory + '/', global_step=epochs_elapsed)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    except tf.errors.CancelledError:
        print('Done training -- epoch limit reached counting checkpoints')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    # And wait for them to actually do it.
    for threads in enqueue_threads:
        coord.join(threads)
    sess.close()
