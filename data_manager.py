import tensorflow as tf
import numpy as np
import time
import os


class DataProcessor(object):

    def __init__(self,
                 root_path='default',
                 name='default',
                 file_num=10,
                 total=1000,
                 dim=[50*50*3, 1],
                 batch=10,
                 epochs=10,
                 shuffle=True):
        self._index = ('data', 'label')
        self._data_set_index = ('train', 'valid', 'test')
        self._file_list = []
        self._root_path = root_path
        self._data_set_name = name
        self._epochs = epochs
        self._shuffle = shuffle
        self._dim = dim

        self._record_file_num = file_num
        self._total = total
        self._each_file_num = int(self._total/self._record_file_num)
        self._batch = batch
        self._batch_num = int(self._total/self._batch)
        self._min_after_dequeue = int(0.8*self._batch_num)

        if not os.path.exists(self._root_path):
            os.mkdir(self._root_path)

        for sub_data_path in self._data_set_index:
            if not os.path.exists(self._root_path + '/' + sub_data_path):
                os.mkdir(self._root_path + '/' + sub_data_path)

            list_tmp = []
            for file_num in range(self._record_file_num):
                list_tmp.append(self._root_path + '/'
                                + sub_data_path + '/'
                                + self._data_set_name
                                + '_%d' % file_num
                                + '.tfrecords')

            self._file_list.append(list_tmp)

        print(self._file_list)
        print(self._file_list[0])
        print('data processor init')

    def _float_feature(self, value):
        print(value)
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    # notice value must be list not numpy array
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value)]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def create_features(self, input, feature_method):
        feature_method(input)

    # data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # label = [1.1, 2.2, 3.3]
    def code_record(self, **input_dict):
        for key, value in input_dict.items():
            if value[0] is 'byte':
                input_dict[key] = self._bytes_feature(value[1:])
            elif value[0] is 'float':
                input_dict[key] = self._float_feature(value[1:])
            elif value[0] is 'int64':
                input_dict[key] = self._int64_feature(value[1:])
            else:
                print('input error')

        features = tf.train.Features(feature=input_dict)
        record = tf.train.Example(features=features)
        return record.SerializeToString()

    def decode_record(self, record):
        features = {
            'data': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([3], tf.float32)
        }

        feature = tf.parse_single_example(record, features=features)
        data = tf.decode_raw(feature['data'], out_type=tf.uint8)
        data = tf.reshape(data, [3])
        label = feature['label']
        return data, label

    def create_tfrecord(self):
        label_tmp = np.array([1])

        for data_set in self._file_list:
            for file in data_set:
                with tf.python_io.TFRecordWriter(file) as writer:
                    for i in range(self._each_file_num):
                        data_tmp = np.random.randint(0, 255, [50 * 50 * 3])

                        features = tf.train.Features(feature={
                            'data': self._bytes_feature(bytes(data_tmp.tolist())),
                            'label': self._bytes_feature(bytes(label_tmp.tolist()))
                        })
                        example = tf.train.Example(features=features)
                        writer.write(example.SerializeToString())
                        # self._data += 1
                    label_tmp += 1

    def tfrecord_decode(self, example):
        features = tf.parse_single_example(example,
                                           features={
                                               'data': tf.FixedLenFeature([], tf.string),
                                               'label': tf.FixedLenFeature([], tf.string)
                                           })

        data = tf.decode_raw(features['data'], out_type=tf.uint8)
        label = tf.decode_raw(features['label'], out_type=tf.uint8)
        return data, label


    def read_tfrecord(self, path):
        files = tf.train.match_filenames_once(path)

        file_queue = tf.train.string_input_producer(files, num_epochs=self._epochs, shuffle=self._shuffle)
        reader = tf.TFRecordReader()
        _, serial_example = reader.read(file_queue)

        data, label = self.tfrecord_decode(serial_example)
        # must reshape a certan shape or tensorflow can not use
        data = tf.reshape(data, [self._dim[0]])
        label = tf.reshape(label, [self._dim[1]])

        data_batch, label_batch = tf.train.shuffle_batch([data, label],
                                                         batch_size=self._batch,
                                                         capacity=self._batch_num,
                                                         min_after_dequeue=self._min_after_dequeue,
                                                         num_threads=self._record_file_num)

        return data_batch, label_batch

    @property
    def train_batch(self):
        return self.read_tfrecord(path=self._root_path + '/train/*.*')

    @property
    def valid_batch(self):
        return self.read_tfrecord(path=self._root_path + '/valid/*.*')

    @property
    def test_batch(self):
        return self.read_tfrecord(path=self._root_path + '/test/*.*')

    # not file must be file_list [file]
    def test_read_record(self, file):
        file_queue = tf.train.string_input_producer(file)
        reader = tf.TFRecordReader()
        _, serial_example = reader.read(file_queue)
        data, label = self.decode_record(serial_example)

        sess = tf.Session()
        sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        d, l = sess.run([data, label])
        coord.request_stop()
        coord.join(threads)
        return d, l


def var_arg(**dict):
    for key, value in dict.items():
        if value[0] is 'byte':
            dict[key] = [1]
        elif value[0] is 'float':
            dict[key] = [2]
    return dict


# tf.image.rot90()

if __name__ == '__main__':
    # dp = DataProcessor()
    # # record = dp.code_record(data=[1, 2, 3], label=[1004.123, 1005.123, 1006.123])
    # # with tf.python_io.TFRecordWriter('test/test.tfrecords') as writer:
    # #     writer.write(record)
    #
    # data, label = dp.test_read_record(['test/test.tfrecords'])
    #
    # print([data, label])
    ret = var_arg(data=['byte', 1, 2, 3], label=['float', 1.2, 3.1, 1.1])
    print(ret)

    # print(['byte', 1, 2, 3])
    # print(['float', 1, 2, 3])
    # label = [1.1, 2.2, 3.3]
    # print(type(label[0]))
    # dp = DataProcessor(root_path='mnist', name='mnist', epochs=1)
    #
    # train_d, train_l = dp.train_batch
    # valid_d, valid_l = dp.valid_batch
    # test_d, test_l = dp.test_batch
    #
    # data, label = dp.test_print_batch()
    #
    # print([data, label])

    # sess = tf.Session()
    # # a trick must init local and global initializer
    # sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
    # # sess.run(tf.local_variables_initializer())
    #
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess, coord=coord)
    #
    # start_t = time.time()
    # count = 0
    # try:
    #     while not coord.should_stop():
    #         # print('train batch:')
    #         d, l = sess.run([train_d, train_l])
    #         print([d, l])
    #         count += 1
    #         # print('valid batch:')
    #         # d, l = sess.run([valid_d, valid_l])
    #         # print([d, l])
    #         #
    #         # print('test batch:')
    #         # d, l = sess.run([test_d, test_l])
    #         # print([d, l])
    # except tf.errors.OutOfRangeError:
    #     print('end epoch')
    # finally:
    #     print('request_stop')
    #     coord.request_stop()
    #
    # end_t = time.time()
    # print('use time %s' % (end_t - start_t))
    # print('count %d'% count)
    # coord.join(threads)

