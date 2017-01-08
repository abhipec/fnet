"""
Our classification model.
"""
import os
import pickle
import tensorflow as tf
#pylint: disable=import-error
import models.modified_hinge_loss

def read_local_variables_and_params(arguments):
    """
    Read other variables and params.
    """
    relevant_directory = os.path.expanduser(
        arguments['--data_directory']) + arguments['--dataset'] + '/'

    l_vars = pickle.load(open(relevant_directory + 'local_variables.pickle', 'rb'))

    #pylint: disable=too-many-instance-attributes,too-few-public-methods
    class Params():
        """
        Parameter class for our model.
        """
        def __init__(self, l_vars, arguments):
            self.output_dim = len(l_vars['num_to_label'])
            self.pre_trained_embedding_shape = l_vars['word_embedding'].shape
            self.char_embedding_shape = (len(l_vars['num_to_chrs']) + 1,
                                         int(arguments['--char_embedding_size']))
            self.rnn_hidden_neurons = int(arguments['--rnn_hidden_neurons'])
            self.char_rnn_hidden_neurons = int(arguments['--char_rnn_hidden_neurons'])
            self.use_mention = arguments['--use_mention']
            self.use_clean = arguments['--use_clean']
            self.embedding_dim = int(arguments['--joint_embedding_size'])
            self.keep_prob = float(arguments['--keep_prob'])
            self.learning_rate = float(arguments['--learning_rate'])
            self.trainable = arguments['--retrain_word_embeddings']
            self.data_size = 0
    params = {}
    params['train'] = Params(l_vars, arguments)
    params['train'].data_size = l_vars['train_size']

    params['dev'] = Params(l_vars, arguments)
    params['dev'].keep_prob = 1
    params['dev'].data_size = l_vars['dev_size']

    params['test'] = Params(l_vars, arguments)
    params['test'].keep_prob = 1
    params['test'].data_size = l_vars['test_size']

    return l_vars, params


def create_placeholders(parameters):
    """
    Create placeholders for data.
    """
    placeholders = {}

    with tf.name_scope('placeholders'):
        placeholders['lci'] = tf.placeholder(tf.int64, [None, None], name='left_context_ids')
        placeholders['emi'] = tf.placeholder(tf.int64, [None, None], name='mention_ids')
        placeholders['rci'] = tf.placeholder(tf.int64, [None, None], name='right_context_ids')
        placeholders['lcl'] = tf.placeholder(tf.int64, [None], name='left_context_length')
        placeholders['eml'] = tf.placeholder(tf.int64, [None], name='mention_length')
        placeholders['rcl'] = tf.placeholder(tf.int64, [None], name='right_context_length')
        placeholders['clean'] = tf.placeholder(tf.bool, [None], name='clean-status')
        placeholders['labels'] = tf.placeholder(tf.float32,
                                                [None, parameters.output_dim],
                                                name='labels')
        placeholders['keep_prob'] = tf.placeholder(tf.float32, name='keep_prob')

    return placeholders
#pylint: disable=too-many-locals
def read_batch(filename, batch_size, random=False):
    """
    Read single batch.
    """
    context_features = {
        "lcl": tf.FixedLenFeature([], dtype=tf.int64),
        "rcl": tf.FixedLenFeature([], dtype=tf.int64),
        "eml": tf.FixedLenFeature([], dtype=tf.int64),
        "clean": tf.FixedLenFeature([], dtype=tf.int64),
        "uid": tf.FixedLenFeature([], dtype=tf.string)}
    sequence_features = {
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "lci": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "rci": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "emi": tf.FixedLenSequenceFeature([], dtype=tf.int64)}

    filename_queue = tf.train.string_input_producer(
        [filename],
        num_epochs=None)

    _, ex = tf.TFRecordReader().read(filename_queue)
    if random:
        # maintain a queue of large capacity
        queue = tf.RandomShuffleQueue(dtypes=[tf.string],
                                      capacity=batch_size * 100,
                                      min_after_dequeue=batch_size * 50)
        enqueue_op = queue.enqueue(ex)
        dequeue_op = queue.dequeue()
        queue_runner = tf.train.QueueRunner(queue, [enqueue_op])
    else:
        dequeue_op = ex

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=dequeue_op,
        context_features=context_features,
        sequence_features=sequence_features
        )

    all_examples = {}
    all_examples.update(context_parsed)
    all_examples.update(sequence_parsed)

    batch = tf.train.batch(
        tensors=all_examples,
        batch_size=batch_size,
        dynamic_pad=True,
        allow_smaller_final_batch=True,
        name="y_batch"
    )
    if random:
        return batch, queue_runner
    else:
        return batch

def read_batches(relevant_directory, batch_size):
    """
    Read train/test/dev batches from tfrecord.
    """
    batches = {}
    qrs = {}
    batches['train'], qrs['train'] = read_batch(relevant_directory + 'train.tfrecord',
                                                batch_size,
                                                random=True)

    batches['test'] = read_batch(relevant_directory + 'test.tfrecord',
                                 batch_size,
                                 random=False)

    batches['dev'] = read_batch(relevant_directory + 'dev.tfrecord',
                                batch_size,
                                random=False)
    batches['size'] = batch_size
    return batches, qrs

#pylint: disable=too-many-locals, too-many-statements
def model(placeholders, parameters, pre_trained_embedding, is_training=False):
    """
    Our classification model.
    """
    reuse = not is_training
    print('Reusing variables:', reuse)
    print(pre_trained_embedding.shape)
    with tf.variable_scope('embeddings', reuse=reuse):
        pre_trained_embedding = tf.get_variable('word_embeddings',
                                                dtype=tf.float32,
                                                initializer=tf.constant(pre_trained_embedding),
                                                trainable=parameters.trainable)

        char_embedding = tf.get_variable('char_embeddings',
                                         initializer=tf.truncated_normal(
                                             parameters.char_embedding_shape,
                                             stddev=0.01),
                                         trainable=True)

        label_embedding = tf.get_variable('label_embeddings',
                                          initializer=tf.truncated_normal(
                                              [parameters.output_dim, parameters.embedding_dim],
                                              stddev=0.01),
                                          trainable=True)

    with tf.name_scope('abhishekClassificationModel'):
        num_neurons = parameters.rnn_hidden_neurons
        cell = tf.nn.rnn_cell.LSTMCell

        with tf.variable_scope('inputs', reuse=reuse):
            left_context = tf.nn.embedding_lookup(pre_trained_embedding,
                                                  placeholders['lci'],
                                                  name='left_context')

            mentions = tf.nn.embedding_lookup(char_embedding,
                                              placeholders['emi'],
                                              name='mention')

            right_context = tf.nn.embedding_lookup(pre_trained_embedding,
                                                   placeholders['rci'],
                                                   name='right_context')

        with tf.variable_scope('mention_projections', reuse=reuse):
            # mhh if output from last cell
            _, (_, mhh) = tf.nn.dynamic_rnn(cell(parameters.char_rnn_hidden_neurons,
                                                 state_is_tuple=True),
                                            mentions,
                                            placeholders['eml'],
                                            dtype=tf.float32,
                                            scope='rnn-mentions')
        with tf.variable_scope('sentence_projections', reuse=reuse):
            # bidirectional encoding of left context
            _, ((_, lcfwh), (_, lcbwh)) = tf.nn.bidirectional_dynamic_rnn(cell(num_neurons,
                                                                               state_is_tuple=True),
                                                                          cell(num_neurons,
                                                                               state_is_tuple=True),
                                                                          left_context,
                                                                          placeholders['lcl'],
                                                                          dtype=tf.float32,
                                                                          scope='lc-bidirectional')
            # concatenate fwd and bwd pass
            lchh = tf.concat(1, [lcfwh, lcbwh])

            # bidirectional encoding of right context
            _, ((_, rcfwh), (_, rcbwh)) = tf.nn.bidirectional_dynamic_rnn(cell(num_neurons,
                                                                               state_is_tuple=True),
                                                                          cell(num_neurons,
                                                                               state_is_tuple=True),
                                                                          right_context,
                                                                          placeholders['rcl'],
                                                                          dtype=tf.float32,
                                                                          scope='rc-bidirectional')
            # concatenate fwd and bwd pass
            rchh = tf.concat(1, [rcfwh, rcbwh])
            # return type (outputs, (hidden_states)), (hidden_states) = (State_c, State_h)
            # State_c = cell state, State_h = cell final output.

        with tf.variable_scope('output_projection', reuse=reuse):
            # combine left and right encoding
            combined = tf.concat(1, [lchh, rchh])
            # apply dropout of left-right combined
            combined_d = tf.nn.dropout(combined, placeholders['keep_prob'], name='dropout_combined')
            # apply dropout on mention representation
            mhhd = tf.nn.dropout(mhh, placeholders['keep_prob'], name='dropout_mention')
            # combine all representations
            if parameters.use_mention:
                all_rep = tf.concat(1, [mhhd, combined_d])
                rep_dim = parameters.char_rnn_hidden_neurons + 4 * parameters.rnn_hidden_neurons
            else:
                all_rep = combined_d
                rep_dim = 4 * parameters.rnn_hidden_neurons

        with tf.variable_scope('score-calculation', reuse=reuse):
            # a matrix of [input_dim, embedding_space] dim.
            matrix_b = tf.get_variable('matrix_b',
                                       initializer=tf.truncated_normal(
                                           [rep_dim, parameters.embedding_dim],
                                           stddev=0.01))
            # a matrix of [batch_size, embedding_space] dim.
            transformation = tf.matmul(all_rep, matrix_b)
            # a matrix of shape [batch_size, no_of_labels]
            scores = tf.matmul(transformation, label_embedding, transpose_b=True)

        with tf.variable_scope('cost_and_optimization', reuse=reuse):
            cost = models.modified_hinge_loss.loss(scores,
                                                   placeholders['labels'],
                                                   placeholders['clean'],
                                                   target_dim=parameters.output_dim,
                                                   use_clean=parameters.use_clean)

            train_op = tf.train.AdamOptimizer(parameters.learning_rate).minimize(cost)

        operations = {}
        operations['prediction'] = scores
        operations['cost'] = cost
        operations['optimize'] = train_op
        operations['representation'] = all_rep
        return operations
