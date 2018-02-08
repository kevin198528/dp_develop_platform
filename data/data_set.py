# from include_pkg import *
# import tensorflow as tf
# import tensorflow.examples.tutorials.mnist as mnist
# import numpy as np
#
# class DataSet(object):
#
#     def __init__(self, file_path):
#         self.name = 'default'
#         pass
#
#     def name(self):
#         pass
#
#     def abstract_data(self):
#         pass
#
#     def abstract_label(self):
#         pass
#
#     def train_dict(self):
#         return {}
#
#     def valid_dict(self):
#         return {}
#
#     def test_dict(self):
#         return {}
#
#
# class LinearData(DataProcess):
#
#     def __init__(self, dim=1, num=100):
#         self._name = 'linear'
#         self._dim = dim
#         self._num = num
#         self._random = np.random.randint(0, num, 1)
#
#         # abstract data
#         self._abs_data = tf.placeholder(dtype=tf.float32, shape=[None, self._dim])
#         # abstract label
#         self._abs_label = tf.placeholder(dtype=tf.float32, shape=[None, 1])
#
#         # real data
#         self._real_data = np.random.randn(100, self._dim)
#         # real label
#         self._real_label = np.dot(self._real_data, [[0.3], [0.1]]) + 0.3
#         print('create linear data')
#
#     def name(self):
#         return self._name
#
#     def abstract_data(self):
#         return self._abs_data
#
#     def abstract_label(self):
#         return self._abs_label
#
#     def train_dict(self):
#         return {self._abs_data: self._real_data,
#                 self._abs_label: self._real_label}
#
#     def valid_dict(self):
#         return {self._abs_data: self._real_data,
#                 self._abs_label: self._real_label}
#
#     def test_dict(self):
#         return {self._abs_data: self._real_data[self._random, :],
#                 self._abs_label: self._real_label[self._random, :]}
#
#
# class MnistData(DataProcess):
#
#     def __init__(self, dim=784):
#         self._name = 'mnist'
#         self._data_path = '../data_set/mnist/'
#         self._raw_mnist_data = mnist.input_data.read_data_sets(self._data_path, one_hot=True)
#         self._data_dim = dim
#         self._label_dim = 10
#         self._latest_epochs = 0
#
#         # abstract data and label
#         self._abs_data = tf.placeholder(dtype=tf.float32, shape=[None, self._data_dim])
#         self._abs_label = tf.placeholder(dtype=tf.float32, shape=[None, self._label_dim])
#
#     @property
#     def latest_epochs(self):
#         self._latest_epochs = self._raw_mnist_data.train.epochs_completed
#         return self._latest_epochs
#
#     def name(self):
#         return self._name
#
#         # real data and label
#     def abstract_data(self):
#         return self._abs_data
#
#     def abstract_label(self):
#         return self._abs_label
#
#     def train_dict(self):
#         train_data, train_label = self._raw_mnist_data.train.next_batch(200)
#         return {self._abs_data: train_data,
#                 self._abs_label: train_label}
#
#     def valid_dict(self):
#         valid_data, valid_label = self._raw_mnist_data.validation.next_batch(1000)
#         return {self._abs_data: valid_data,
#                 self._abs_label: valid_label}
#
#     def test_dict(self):
#         test_data, test_label = self._raw_mnist_data.test.next_batch(1000)
#         return {self._abs_data: test_data,
#                 self._abs_label: test_label}
#
#
# if __name__ == '__main__':
#     mnist_data = MnistData()
#     print(mnist_data._raw_mnist_data.train.images.shape)
#     print(mnist_data._raw_mnist_data.train.labels.shape)
#     print(mnist_data._raw_mnist_data.validation.images.shape)
#     print(mnist_data._raw_mnist_data.validation.labels.shape)
#     print(mnist_data._raw_mnist_data.test.images.shape)
#     print(mnist_data._raw_mnist_data.test.labels.shape)
# from include_pkg import *
# import tensorflow as tf
# import tensorflow.examples.tutorials.mnist as mnist
# import numpy as np
#
# class DataProcess(object):
#
#     def __init__(self):
#         self.name = 'default'
#         pass
#
#     def name(self):
#         pass
#
#     def abstract_data(self):
#         pass
#
#     def abstract_label(self):
#         pass
#
#     def train_dict(self):
#         return {}
#
#     def valid_dict(self):
#         return {}
#
#     def test_dict(self):
#         return {}
#
#
# class LinearData(DataProcess):
#
#     def __init__(self, dim=1, num=100):
#         self._name = 'linear'
#         self._dim = dim
#         self._num = num
#         self._random = np.random.randint(0, num, 1)
#
#         # abstract data
#         self._abs_data = tf.placeholder(dtype=tf.float32, shape=[None, self._dim])
#         # abstract label
#         self._abs_label = tf.placeholder(dtype=tf.float32, shape=[None, 1])
#
#         # real data
#         self._real_data = np.random.randn(100, self._dim)
#         # real label
#         self._real_label = np.dot(self._real_data, [[0.3], [0.1]]) + 0.3
#         print('create linear data')
#
#     def name(self):
#         return self._name
#
#     def abstract_data(self):
#         return self._abs_data
#
#     def abstract_label(self):
#         return self._abs_label
#
#     def train_dict(self):
#         return {self._abs_data: self._real_data,
#                 self._abs_label: self._real_label}
#
#     def valid_dict(self):
#         return {self._abs_data: self._real_data,
#                 self._abs_label: self._real_label}
#
#     def test_dict(self):
#         return {self._abs_data: self._real_data[self._random, :],
#                 self._abs_label: self._real_label[self._random, :]}
#
#
# class MnistData(DataProcess):
#
#     def __init__(self, dim=784):
#         self._name = 'mnist'
#         self._data_path = '../data_set/mnist/'
#         self._raw_mnist_data = mnist.input_data.read_data_sets(self._data_path, one_hot=True)
#         self._data_dim = dim
#         self._label_dim = 10
#         self._latest_epochs = 0
#
#         # abstract data and label
#         self._abs_data = tf.placeholder(dtype=tf.float32, shape=[None, self._data_dim])
#         self._abs_label = tf.placeholder(dtype=tf.float32, shape=[None, self._label_dim])
#
#     @property
#     def latest_epochs(self):
#         self._latest_epochs = self._raw_mnist_data.train.epochs_completed
#         return self._latest_epochs
#
#     def name(self):
#         return self._name
#
#         # real data and label
#     def abstract_data(self):
#         return self._abs_data
#
#     def abstract_label(self):
#         return self._abs_label
#
#     def train_dict(self):
#         train_data, train_label = self._raw_mnist_data.train.next_batch(200)
#         return {self._abs_data: train_data,
#                 self._abs_label: train_label}
#
#     def valid_dict(self):
#         valid_data, valid_label = self._raw_mnist_data.validation.next_batch(1000)
#         return {self._abs_data: valid_data,
#                 self._abs_label: valid_label}
#
#     def test_dict(self):
#         test_data, test_label = self._raw_mnist_data.test.next_batch(1000)
#         return {self._abs_data: test_data,
#                 self._abs_label: test_label}
#
#
# if __name__ == '__main__':
#     mnist_data = MnistData()
#     print(mnist_data._raw_mnist_data.train.images.shape)
#     print(mnist_data._raw_mnist_data.train.labels.shape)
#     print(mnist_data._raw_mnist_data.validation.images.shape)
#     print(mnist_data._raw_mnist_data.validation.labels.shape)
#     print(mnist_data._raw_mnist_data.test.images.shape)
#     print(mnist_data._raw_mnist_data.test.labels.shape)
