from include_pkg import *
import pickle
from data_process import *
from nn_layer import *
from loss_method import *
from deep_learning_frame import *

class SuperParam(object):

    # param agument is object type
    def __init__(self, data, net, loss):
        self._data = data
        self._net = net
        self._loss = loss
        pass

    def data(self):
        return self._data

    def net(self):
        return self._net

    def loss(self):
        return self._loss

param = SuperParam(MnistData(), MniConvNet(), MnistLoss(loss_dim=10))

dp = DeepLearningFrame(param.data(), param.net(), param.loss())

# dp = DeepLearningFrame(MnistData(), MniConvNet(), MnistLoss(loss_dim=10))

dp.train(learn_rat=0.00003, epoch=100, show=100)

dp.test()

#
# _num_examples = 3
#
# _images = np.array([[[1]], [[2]], [[3]]])
#
# perm = np.arange(_num_examples)
# print(perm)
# np.random.shuffle(perm)
# print(perm)
#
# _images = _images[perm]
# # _labels = _labels[perm]
#
# print(_images)


# def label_to_hot(label):
#     tmp = np.arange(label.max() + 1)
#     return (tmp == label[:, None]).astype(np.int)
#
# def conver_batch_to_hot(path, save_file):
#     with open(path, "rb+") as f:
#         dict_data = pickle.load(f, encoding='bytes')
#         # for key, value in dict_data.items():
#         #     print(key)
#
#     data = np.array(dict_data[b'data'])
#     labels = np.array(dict_data[b'labels'])
#
#     hot_labels = label_to_hot(labels)
#
#     with open(save_file, 'wb+') as f:
#         pickle.dump({'data': data, 'labels': hot_labels}, f)
#
# def load_bath(path):
#     with open(path, "rb+") as f:
#         dict_data = pickle.load(f, encoding='bytes')
#
#     for key, value in dict_data.items():
#         print(key)
#
#     print(dict_data['data'].shape)
#     # print(dict_data['data'][0:10])
#     # print(dict_data['labels'][0:10])
#
# def demo(path):
#     # --- 序列化 ---
#     # f = open("pickle.txt", "wb+")
#     # lists = [123, "中文", [456]]
#     # strs = "字符串"
#     # num = 123
#     #
#     # # 写入
#     # pickle.dump(lists, f) # 序列化到文件
#     # pickle.dump(strs, f)
#     # pickle.dump(num, f)
#     #
#     # # 关闭
#     # f.close()
#
#
#     # --- 反序列化 ---
#     f = open(path, "rb+")
#
#     # 读取
#     dict_data = pickle.load(f, encoding='bytes') # 从文件反序列化
#     # print(dict_data['data'])
#     # print(dict_data['labels'])
#     for key, value in dict_data.items():
#         print(key)
#
#     data = np.array(dict_data[b'data'])
#
#     labels = np.array(dict_data[b'labels'])
#
#     # print(data.shape)
#     print(labels[0:10])
#     #
#     # print(data_labels[0:20])
#     # pk_data = data_datas[0:10]
#     # pk_label = label_to_hot(data_labels[0:10])
#
#     f.close()
#
#     # fw = open("pickle_batch_10", "wb+")
#     #
#     # d = {'data':pk_data, 'label':pk_label}
#     # pickle.dump(d, fw)
#     # fw.close()
#
# path = '../data_set/cifar-10/data_batch_1'
# s_path = '../data_set/cifar-10/data_batch_1_hot'
# # conver_batch_to_hot(path)
#
# # conver_batch_to_hot(path, s_path)
#
# # load_bath(s_path)
# # demo(path)
#
#
# from collections import namedtuple
#
# dataset = namedtuple('d', ['train', 'valid', 'test'])
#
# data = dataset(train=1, valid=2, test=3)
#
# print(data)


# directory = './mnist_ckpt/*.*'
# file_names = tf.train.match_filenames_once(directory)
#
# init = (tf.global_variables_initializer(), tf.local_variables_initializer())
# print('1')
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     print(sess.run(file_names))
#
# tf.train.string_input_producer()
#
# tf.train.shuffle_batch()

# data = tf.ones([1, 4, 4, 2], dtype=tf.float32)
#
# f = tf.ones([2, 2, 2, 2], dtype=tf.float32)
#
# feature = tf.nn.conv2d(data, f, [1, 2, 2, 1], padding='SAME')
#
# ttu.print_var(feature)

import numpy as np

# data = tf.constant([[[0.8, 0.6], [0.1, 0.3], [0.5, 0.1]], [[0.8, 0.6], [0.1, 0.3], [0.5, 0.1]]])
#
# label = tf.constant([[[0.0, 0.6], [0.1, 0.3], [0.5, 0.1]], [[0.0, 0.6], [0.1, 0.3], [0.5, 0.1]]])
# data = tf.reshape(data, shape=[12])
# label = tf.reshape(label, shape=[12])
# # def distribut_cross_entropy(data, label):
#
#
# ret = (-tf.log(1 - tf.abs(data - label)))
#
# a = tf.constant(0.5, dtype=tf.float32)

# a = tf.constant([1, 2, 3])
# b = tf.constant([4, 5, 6])
#
# ttu.print_var(a*b)


