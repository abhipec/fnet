"""
Our classification model.
"""
import os
import pickle
import tensorflow as tf
#pylint: disable=import-error
import models.custom_warp_loss

def read_local_variables_and_params(arguments):
    """
    Read other variables and params.
    """
    relevant_directory = os.path.expanduser(
        arguments['--data_directory']) + arguments['--dataset'] + '/'

    representations = pickle.load(open(relevant_directory + 'all_representations.pickle', 'rb'))

    #pylint: disable=too-many-instance-attributes,too-few-public-methods
    class Params():
        """
        Parameter class for our model.
        """
        def __init__(self, representation, arguments):
            self.output_dim = len(representation['num_to_label'])
            self.feature_dim = representation['dev']['features'].shape[1]
            self.embedding_dim = int(arguments['--joint_embedding_size'])
            self.learning_rate = float(arguments['--learning_rate'])
            self.use_clean = arguments['--use_clean']
    params = {}
    params['train'] = Params(representations, arguments)
    params['dev'] = Params(representations, arguments)
    params['test'] = Params(representations, arguments)
    return representations, params


def create_placeholders(parameters):
    """
    Create placeholders for data.
    """
    placeholders = {}

    with tf.name_scope('placeholders'):
        placeholders['features'] = tf.placeholder(tf.float32,
                                                  [None, parameters.feature_dim],
                                                  name='features')
        placeholders['clean'] = tf.placeholder(tf.bool, [None], name='clean-status')
        placeholders['labels'] = tf.placeholder(tf.float32,
                                                [None, parameters.output_dim],
                                                name='labels')
    return placeholders

#pylint: disable=too-many-locals, too-many-statements
def model(placeholders, parameters, correlation_matrix, is_training=False):
    """
    Our classification model.
    """
    reuse = not is_training
    print('Reusing variables:', reuse)
    with tf.variable_scope('embeddings', reuse=reuse):
        cmatrix = tf.get_variable('correlation_matrix',
                                  dtype=tf.float32,
                                  initializer=tf.constant(correlation_matrix),
                                  trainable=False)

        label_embedding = tf.get_variable('label_embeddings',
                                          initializer=tf.truncated_normal(
                                              [parameters.output_dim, parameters.embedding_dim],
                                              stddev=0.01),
                                          trainable=True)

    with tf.name_scope('TransferLearningModel'):
        with tf.variable_scope('score-calculation', reuse=reuse):
            # a matrix of [input_dim, embedding_space] dim.
            matrix_b = tf.get_variable('matrix_b',
                                       initializer=tf.truncated_normal(
                                           [parameters.feature_dim, parameters.embedding_dim],
                                           stddev=0.01))
            # a matrix of [batch_size, embedding_space] dim.
            transformation = tf.matmul(placeholders['features'], matrix_b)

            #normalized_transformation = tf.nn.l2_normalize(transformation, 1)
            #normalized_label_embedding = tf.nn.l2_normalize(label_embedding, 1)
            # a matrix of shape [batch_size, no_of_labels]
            scores = tf.matmul(transformation,
                               label_embedding,
                               transpose_b=True)

        with tf.variable_scope('cost_and_optimization', reuse=reuse):
            cost = models.custom_warp_loss.loss(scores,
                                                placeholders['labels'],
                                                cmatrix,
                                                placeholders['clean'],
                                                parameters.output_dim,
                                                use_clean=parameters.use_clean)

            train_op = tf.train.AdamOptimizer(parameters.learning_rate).minimize(cost)

        operations = {}
        operations['prediction'] = scores
        operations['cost'] = cost
        operations['optimize'] = train_op
        return operations
