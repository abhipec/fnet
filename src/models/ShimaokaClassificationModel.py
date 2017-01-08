"""
Our classification model.
"""
import os
import pickle
import tensorflow as tf

#pylint: disable=too-many-locals
def attention(inputs,
              input_shape,
              attention_size,
              context=None,
              context_shape=None):
    """General attention mechanism model.

    It return the weighted average of inputs and the weights.

    Args:
        inputs: A 3D shaped Tensor [batch_size x time x input_size].
        attention_size: Size of attention unit.
        context: A 2D shaped Tensor [batch_size x context_size].

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A 2D shaped Tensor [batch_size x input_size].
            states: A 2D shaped Tensor [batch_size x time].

    """

    with tf.name_scope("attention_mechanism"):

        _, time, input_size = input_shape

        if context is not None:
            _, context_size = context_shape

        with tf.name_scope('weights'):
            # First level
            weight_1 = tf.Variable(tf.truncated_normal([input_size, attention_size], stddev=0.01))
            # Second level
            weight_2 = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.01))

            if context is not None:
                weight_3 = tf.Variable(tf.truncated_normal([context_size, attention_size],
                                                           stddev=0.01))

        with tf.name_scope('projections'):
            if context is not None:
                #apply first level attention, output shape will be(batch_size, time, attention_size)
                first_level_sequence = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, input_size]),
                                                            weight_1),
                                                  [-1, time, attention_size])
                first_level_context = tf.expand_dims(tf.matmul(context, weight_3), 1)
                first_level = tf.reshape(tf.tanh(first_level_sequence + first_level_context),
                                         [-1, attention_size])
                unnormalized_weights = tf.matmul(first_level, weight_3)
            else:
                first_level = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, input_size]), weight_1))
                unnormalized_weights = tf.matmul(first_level, weight_2)

        with tf.name_scope('outputs'):
            # normalize outputs
            attention_weights = tf.nn.softmax(tf.reshape(unnormalized_weights, [-1, time]))
            out = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, 2), reduction_indices=1)

    return out, attention_weights

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
            self.rnn_hidden_neurons = int(arguments['--rnn_hidden_neurons'])
            self.keep_prob = float(arguments['--keep_prob'])
            self.learning_rate = float(arguments['--learning_rate'])
            self.attention_size = int(arguments['--attention_size'])
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

    with tf.name_scope('ShimaokaClassificationModel'):
        num_neurons = parameters.rnn_hidden_neurons
        cell = tf.nn.rnn_cell.LSTMCell

        with tf.variable_scope('inputs', reuse=reuse):
            left_context = tf.nn.embedding_lookup(pre_trained_embedding,
                                                  placeholders['lci'],
                                                  name='left_context')

            mentions = tf.nn.embedding_lookup(pre_trained_embedding,
                                              placeholders['emi'],
                                              name='mention')

            right_context = tf.nn.embedding_lookup(pre_trained_embedding,
                                                   placeholders['rci'],
                                                   name='right_context')

        with tf.variable_scope('mention_projections', reuse=reuse):
            m_average = tf.reduce_sum(mentions, reduction_indices=1) / tf.expand_dims(tf.cast(placeholders['eml'],
                                                                                              tf.float32), 1)

        with tf.variable_scope('sentence_projections', reuse=reuse):
            # bidirectional encoding of left context
            lcos, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell(num_neurons, state_is_tuple=True, use_peepholes=True),
                cell_bw=cell(num_neurons, state_is_tuple=True, use_peepholes=True),
                inputs=left_context,
                sequence_length=placeholders['lcl'],
                dtype=tf.float32,
                scope='lc-bidirectional')
            lco = tf.concat(2, lcos)

            # bidirectional encoding of right context
            rcos, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell(num_neurons, state_is_tuple=True, use_peepholes=True),
                cell_bw=cell(num_neurons, state_is_tuple=True, use_peepholes=True),
                inputs=right_context,
                sequence_length=placeholders['rcl'],
                dtype=tf.float32,
                scope='rc-bidirectional')
            rco = tf.concat(2, rcos)
            # return type (outputs, (hidden_states)), (hidden_states) = (State_c, State_h)
            # State_c = cell state, State_h = cell final output.
            combined, _ = attention(tf.concat(1, (lco, rco)),
                                    (None, 30, 2 * parameters.rnn_hidden_neurons),
                                    parameters.attention_size)

        with tf.variable_scope('output_projection', reuse=reuse):
            # apply dropout on mention representation
            mhhd = tf.nn.dropout(m_average, placeholders['keep_prob'], name='dropout_mention')
            # combine all representations
            all_rep = tf.concat(1, [mhhd, combined])

        with tf.variable_scope('prediction', reuse=reuse):
            # a matrix of [input_dim, embedding_space] dim.
            weights = tf.get_variable(
                'final_weights',
                initializer=tf.truncated_normal(
                    [parameters.pre_trained_embedding_shape[1] + 2 * num_neurons, parameters.output_dim],
                    stddev=0.01))

            y_predict = tf.matmul(all_rep, weights, name='predicted_output')

        with tf.variable_scope('cost_and_optimization', reuse=reuse):
            cost = tf.reduce_mean(
                tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(y_predict,
                                                                      placeholders['labels']),
                              reduction_indices=1))

            train_op = tf.train.AdamOptimizer(parameters.learning_rate).minimize(cost)

        operations = {}
        operations['prediction'] = tf.sigmoid(y_predict)
        operations['cost'] = cost
        operations['optimize'] = train_op
        operations['representation'] = all_rep
        return operations
