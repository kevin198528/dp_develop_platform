from include_pkg import *
import tensorflow as tf


class NeuralNet(object):

    def __init__(self):
        pass

    def weight_variable(self, shape, std=0.1):
        w_init = tf.random_normal(shape, stddev=std)
        return tf.Variable(w_init, dtype=tf.float32)

    def bias_variable(self, shape, std=0.05):
        b_init = tf.random_normal(shape, stddev=std)
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
        self._w_convA1 = self.weight_variable([3, 3, 1, 16], std=0.16)
        self._b_convA1 = self.bias_variable([16], std=0.16)

        self._w_convA2 = self.weight_variable([3, 3, 16, 16], std=0.12)
        self._b_convA2 = self.bias_variable([16], std=0.1)
        # **************************
        self._w_convB1 = self.weight_variable([3, 3, 16, 32], std=0.12)
        self._b_convB1 = self.bias_variable([32], std=0.1)

        self._w_convB2 = self.weight_variable([3, 3, 32, 32], std=0.06)
        self._b_convB2 = self.bias_variable([32], std=0.06)
        # **************************

        # self._w_convC1 = self.weight_variable([3, 3, 16, 32])
        # self._b_convC1 = self.bias_variable([32])
        #
        # self._w_convC2 = self.weight_variable([3, 3, 32, 32])
        # self._b_convC2 = self.bias_variable([32])
        # **************************
        self._w_convD = self.weight_variable([7, 7, 32, 10], std=0.06)
        self._b_convD = self.bias_variable([10], std=0.03)

        # self._w_fc1 = self.weight_variable([7*7*64, 1024])
        # self._b_fc1 = self.weight_variable([1024])
        #
        # self._w_fc2 = self.weight_variable([1024, 10])
        # self._b_fc2 = self.weight_variable([10])

        self._keep_prob = tf.placeholder("float")

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            # 计算参数的均值，并使用tf.summary.scaler记录
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)

            # 计算参数的标准差
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
            tf.summary.scalar('stddev', stddev)
            # tf.summary.scalar('max', tf.reduce_max(var))
            # tf.summary.scalar('min', tf.reduce_min(var))
            # 用直方图记录参数的分布
            # tf.summary.histogram('histogram', var)

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

        with tf.name_scope('layer_conv_A1'):
            with tf.name_scope('weight'):
                self.variable_summaries(self._w_convA1)
            with tf.name_scope('bias'):
                self.variable_summaries(self._b_convA1)

        with tf.name_scope('layer_conv_A2'):
            with tf.name_scope('weight'):
                self.variable_summaries(self._w_convA2)
            with tf.name_scope('bias'):
                self.variable_summaries(self._b_convA2)

        with tf.name_scope('A2_out_put'):
                self.variable_summaries(h_poolA)

        h_convB1 = tf.nn.relu(self.conv2d(h_poolA, self._w_convB1) + self._b_convB1)
        h_convB2 = tf.nn.relu(self.conv2d(h_convB1, self._w_convB2) + self._b_convB2)
        h_poolB = self.max_pool_2x2(h_convB2)

        with tf.name_scope('layer_conv_B1'):
            with tf.name_scope('weight'):
                self.variable_summaries(self._w_convB1)
            with tf.name_scope('bias'):
                self.variable_summaries(self._b_convB1)

        with tf.name_scope('layer_conv_B2'):
            with tf.name_scope('weight'):
                self.variable_summaries(self._w_convB2)
            with tf.name_scope('bias'):
                self.variable_summaries(self._b_convB2)

        with tf.name_scope('B2_out_put'):
                self.variable_summaries(h_poolA)


        # h_fc2 = tf.nn.relu(self.conv2d(h_poolB, self._w_convC1) + self._b_convC1)
        h_fc2 = tf.nn.relu(tf.nn.conv2d(h_poolB, self._w_convD, strides=[1, 1, 1, 1], padding='VALID')
                           + self._b_convD)

        with tf.name_scope('layer_conv_D'):
            with tf.name_scope('weight'):
                self.variable_summaries(self._w_convD)
            with tf.name_scope('bias'):
                self.variable_summaries(self._b_convD)

        with tf.name_scope('D_out_put'):
                self.variable_summaries(h_fc2)

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


class MniConvNet_BN(NeuralNet):

    def __init__(self):
        self._name = 'conv'
        self._w_convA1 = self.weight_variable([3, 3, 1, 16], std=0.16)
        self._b_convA1 = self.bias_variable([16], std=0.16)

        self._w_convA2 = self.weight_variable([3, 3, 16, 16], std=0.12)
        self._b_convA2 = self.bias_variable([16], std=0.1)
        # **************************
        self._w_convB1 = self.weight_variable([3, 3, 16, 32], std=0.12)
        self._b_convB1 = self.bias_variable([32], std=0.1)

        self._w_convB2 = self.weight_variable([3, 3, 32, 32], std=0.06)
        self._b_convB2 = self.bias_variable([32], std=0.06)
        # **************************

        # self._w_convC1 = self.weight_variable([3, 3, 16, 32])
        # self._b_convC1 = self.bias_variable([32])
        #
        # self._w_convC2 = self.weight_variable([3, 3, 32, 32])
        # self._b_convC2 = self.bias_variable([32])
        # **************************
        self._w_convD = self.weight_variable([7, 7, 32, 10], std=0.06)
        self._b_convD = self.bias_variable([10], std=0.03)

        # self._w_fc1 = self.weight_variable([7*7*64, 1024])
        # self._b_fc1 = self.weight_variable([1024])
        #
        # self._w_fc2 = self.weight_variable([1024, 10])
        # self._b_fc2 = self.weight_variable([10])

        self._keep_prob = tf.placeholder("float")

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            # 计算参数的均值，并使用tf.summary.scaler记录
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)

            # 计算参数的标准差
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
            tf.summary.scalar('stddev', stddev)
            # tf.summary.scalar('max', tf.reduce_max(var))
            # tf.summary.scalar('min', tf.reduce_min(var))
            # 用直方图记录参数的分布
            # tf.summary.histogram('histogram', var)

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

        with tf.name_scope('layer_conv_A1'):
            with tf.name_scope('weight'):
                self.variable_summaries(self._w_convA1)
            with tf.name_scope('bias'):
                self.variable_summaries(self._b_convA1)

        with tf.name_scope('layer_conv_A2'):
            with tf.name_scope('weight'):
                self.variable_summaries(self._w_convA2)
            with tf.name_scope('bias'):
                self.variable_summaries(self._b_convA2)

        with tf.name_scope('A2_out_put'):
                self.variable_summaries(h_poolA)

        h_convB1 = tf.nn.relu(self.conv2d(h_poolA, self._w_convB1) + self._b_convB1)
        h_convB2 = tf.nn.relu(self.conv2d(h_convB1, self._w_convB2) + self._b_convB2)
        h_poolB = self.max_pool_2x2(h_convB2)

        with tf.name_scope('layer_conv_B1'):
            with tf.name_scope('weight'):
                self.variable_summaries(self._w_convB1)
            with tf.name_scope('bias'):
                self.variable_summaries(self._b_convB1)

        with tf.name_scope('layer_conv_B2'):
            with tf.name_scope('weight'):
                self.variable_summaries(self._w_convB2)
            with tf.name_scope('bias'):
                self.variable_summaries(self._b_convB2)

        with tf.name_scope('B2_out_put'):
                self.variable_summaries(h_poolA)


        # h_fc2 = tf.nn.relu(self.conv2d(h_poolB, self._w_convC1) + self._b_convC1)
        h_fc2 = tf.nn.relu(tf.nn.conv2d(h_poolB, self._w_convD, strides=[1, 1, 1, 1], padding='VALID')
                           + self._b_convD)

        with tf.name_scope('layer_conv_D'):
            with tf.name_scope('weight'):
                self.variable_summaries(self._w_convD)
            with tf.name_scope('bias'):
                self.variable_summaries(self._b_convD)

        with tf.name_scope('D_out_put'):
                self.variable_summaries(h_fc2)

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