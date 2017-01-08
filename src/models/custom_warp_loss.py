"""
Helper functions.
"""
import time
import numpy as np
from scipy.optimize import check_grad
import tensorflow as tf
from tensorflow.python.framework import ops

#pylint: disable-msg=C0103
loss_module = tf.load_op_library('lib/warp_loss.so')
loss_grad_module = tf.load_op_library('lib/warp_loss_grad.so')

@ops.RegisterGradient("WarpLoss")
def _warp_loss_grad(op, grad):
    """
    The gradients for `zero_out`.
    Args:
        op: The `zero_out` `Operation` that we are differentiating, which we can use
            to find the inputs and outputs of the original op.
            grad: Gradient with respect to the output of the `zero_out` op.
    Returns:
        Gradients with respect to the input of `zero_out`.
    """
    scores = op.inputs[0]
    targets = op.inputs[1]
    correlation_matrix = op.inputs[2]
    clean = op.inputs[3]
    target_dim = op.get_attr('target_dim')
    use_clean = op.get_attr('use_clean')

    gradIn1, gradIn2, gradIn3, gradIn4 = loss_grad_module.warp_loss_grad(scores,
                                                                         targets,
                                                                         correlation_matrix,
                                                                         clean,
                                                                         target_dim=target_dim,
                                                                         use_clean=use_clean)
    gradOut = tf.mul(grad, gradIn1)
    return [gradOut, gradIn2, gradIn3, gradIn4]  # List of one Tensor, since we have one input
#pylint: disable=too-many-arguments
def loss(scores, targets, correlation_matrix, clean, target_dim, use_clean):
    """
    Loss function.
    """

    cost = loss_module.warp_loss(scores,
                                 targets,
                                 correlation_matrix,
                                 clean,
                                 target_dim=target_dim,
                                 use_clean=use_clean)
    return cost


def grad_test():
    """
    A gradient testing function.
    Testing will fail for kinks. (when label is 0 and score is -1).
    """
    scores = np.array([
        [8, 0, 2, 1, -8],
        [-10, -3, 7, -8, 7],
        [-7, -2, 6, -2, 8],
        [8, -9, -3, -1, 6],
        [-7, -4, -6, 1, 5],
        [-4, -6, -9, 4, 1]], dtype=np.core.numerictypes.float64)

    scores = np.reshape(scores, -1)

    targets = tf.Variable(np.array([
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1]]), dtype=tf.float64, trainable=False)

    clean = tf.Variable(np.array([True, False, True, False, True, False]),
                        dtype=tf.bool,
                        trainable=False)
    target_dim = 5
    correlation_matrix = np.ones((target_dim, target_dim))
    use_clean = False

    sess = tf.Session()
    def test_f(scores, sess):
        """
        Test function.
        """
        scores2 = np.reshape(scores, (-1, target_dim))
        scores1 = tf.Variable(scores2)
        cost = loss_module.warp_loss(scores1,
                                     targets,
                                     correlation_matrix,
                                     clean,
                                     target_dim,
                                     use_clean)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        return sess.run(cost)

    def test_g(scores, sess):
        """
        Test gradient.
        """
        scores2 = np.reshape(scores, (-1, target_dim))
        scores1 = tf.Variable(scores2)
        cost = loss_module.warp_loss(scores1,
                                     targets,
                                     correlation_matrix,
                                     clean,
                                     target_dim,
                                     use_clean)
        var_grad = tf.gradients(cost, [scores1])[0]
        sess.run(tf.initialize_all_variables())
        grad = sess.run(var_grad)
        grad = np.reshape(grad, -1)
        return grad

    keywords = {'epsilon': 0.01}
    err = check_grad(test_f, test_g, scores, sess, **keywords)
    time.sleep(1)
    err = np.linalg.norm(err)
    print('Gradient testing error =', err)
    assert err < 1e-6, \
        "Gradient testing error"


def convergence_test():
    """
    Convergence testing function.
    """
    with tf.device('/cpu:0'):
        scores = tf.Variable(np.array([
            [8, 0, 2, 1, -8],
            [-10, -1, 7, -8, 7],
            [-7, -1, 6, -1, 8],
            [8, -9, -3, -1, 6],
            [-7, -4, -6, 1, 5],
            [-4, -6, -9, 4, -1]]), dtype=tf.float64)

        targets = tf.Variable(np.array([
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1]]), dtype=tf.float64, trainable=False)

        clean = tf.Variable(np.array([True, False, True, False, True, False]),
                            dtype=tf.bool,
                            trainable=False)
        correlation_matrix = tf.Variable(np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]]), dtype=tf.float64)

        cost = loss(scores, targets, correlation_matrix, clean, target_dim=5, use_clean=False)
    optimize = tf.train.AdamOptimizer(0.1).minimize(cost)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for i in range(100):
        print(i, sess.run(cost))
        _ = sess.run(optimize)

if __name__ == '__main__':
    convergence_test()
    grad_test()
