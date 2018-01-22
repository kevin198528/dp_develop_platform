from include_pkg import *
import tensorflow as tf


class NeuralNet(object):

    def __init__(self):
        pass

    def weight_variable(self, shape):
        w_init = tf.random_normal(shape, stddev=0.15)
        return tf.Variable(w_init, dtype=tf.float32)

    def bias_variable(self, shape):
        b_init = tf.random_normal(shape, stddev=0.05)
        return tf.Variable(b_init, dtype=tf.float32)

    def forward_pass(self, data_in):
        pass

    def get_weights(self):
        pass

    def train_dict(self):
        return {}

    def valid_dict(self):
        return {}

    def test_dict(self):
        return {}

class LinearNet(NeuralNet):

    def __init__(self, shape=[2, 1]):
        self._name = 'linear'
        # w_init = tf.random_normal([input_dim, output_dim], stddev=0.1)
        # self._W = tf.Variable(w_init, dtype=tf.float32)
        # self._B = tf.Variable([output_dim], dtype=tf.float32)
        self._W = self.weight_variable(shape)
        self._B = self.bias_variable([shape[1]])

    def name(self):
        return self._name

    def forward_pass(self, data_in):
        return tf.matmul(data_in, self._W) + self._B

    def get_weights(self):
        return self._W, self._B


class MniConvNet(NeuralNet):

    def __init__(self):
        self._name = 'conv'
        self._w_convA1 = self.weight_variable([3, 3, 1, 8])
        self._b_convA1 = self.bias_variable([8])

        self._w_convA2 = self.weight_variable([3, 3, 8, 8])
        self._b_convA2 = self.bias_variable([8])
        # **************************
        self._w_convB1 = self.weight_variable([3, 3, 8, 16])
        self._b_convB1 = self.bias_variable([16])

        self._w_convB2 = self.weight_variable([3, 3, 16, 16])
        self._b_convB2 = self.bias_variable([16])
        # **************************

        self._w_convC1 = self.weight_variable([3, 3, 16, 32])
        self._b_convC1 = self.bias_variable([32])

        self._w_convC2 = self.weight_variable([3, 3, 32, 32])
        self._b_convC2 = self.bias_variable([32])
        # **************************
        self._w_convD = self.weight_variable([4, 4, 32, 10])
        self._b_convD = self.bias_variable([10])

        # self._w_fc1 = self.weight_variable([7*7*64, 1024])
        # self._b_fc1 = self.weight_variable([1024])
        #
        # self._w_fc2 = self.weight_variable([1024, 10])
        # self._b_fc2 = self.weight_variable([10])

        self._keep_prob = tf.placeholder("float")

    def name(self):
        return self._name

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def forward_pass(self, data_in):
        x_image = tf.reshape(data_in, [-1, 28, 28, 1])
        h_convA1 = tf.nn.relu(self.conv2d(x_image, self._w_convA1) + self._b_convA1)
        h_convA2 = tf.nn.relu(self.conv2d(h_convA1, self._w_convA2) + self._b_convA2)
        h_poolA = self.max_pool_2x2(h_convA2)

        h_convB1 = tf.nn.relu(self.conv2d(h_poolA, self._w_convB1) + self._b_convB1)
        h_convB2 = tf.nn.relu(self.conv2d(h_convB1, self._w_convB2) + self._b_convB2)
        h_poolB = self.max_pool_2x2(h_convB2)

        h_convC1 = tf.nn.relu(self.conv2d(h_poolB, self._w_convC1) + self._b_convC1)
        h_convC2 = tf.nn.relu(self.conv2d(h_convC1, self._w_convC2) + self._b_convC2)
        h_poolC = self.max_pool_2x2(h_convC2)

        # h_fc2 = tf.nn.relu(self.conv2d(h_poolB, self._w_convC1) + self._b_convC1)
        h_fc2 = tf.nn.relu(tf.nn.conv2d(h_poolC, self._w_convD, strides=[1, 1, 1, 1], padding='VALID')
                           + self._b_convD)

        ret = tf.reshape(h_fc2, shape=[-1, 10])

        # h_pool2_flat = tf.reshape(h_pool, [-1, 7 * 7 * 64])
        # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self._w_fc1) + self._b_fc1)
        #
        # h_fc1_drop = tf.nn.dropout(h_fc1, self._keep_prob)
        #
        # h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, self._w_fc2) + self._b_fc2)
        return ret
        # return h_poolB

    def train_dict(self):
        return {self._keep_prob: 1}

    def valid_dict(self):
        return {self._keep_prob: 1}

    def test_dict(self):
        return {self._keep_prob: 1}
