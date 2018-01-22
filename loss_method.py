from include_pkg import *
import tensorflow as tf
import tf_test_utils as ttu

class LossMethod(object):

    def __init__(self, loss_dim=1):
        self._loss_dim = loss_dim
        self._w_loss = self.loss_weight_var()
        self._hard_num = tf.placeholder(dtype=tf.float32, shape=[1])

    def loss_weight_var(self):
        # w_init = tf.random_uniform([self._loss_dim], 0, 1)
        w_init = tf.truncated_normal([self._loss_dim], mean=0, stddev=1.0)
        return tf.Variable(w_init, dtype=tf.float32)

    def machine(self, data, label):
        pass

    def human(self, data, label):
        pass

class L1Loss(LossMethod):

    def __init__(self):
        print('create l1 loss method')

    def machine(self, data, label):
        return tf.reduce_mean(tf.abs(data - label))

    def human(self, data, label):
        return tf.reduce_mean(tf.abs(data - label))


class LinearLoss(LossMethod):

    def __init__(self):
        print('create l2 loss method')

    def machine(self, data, label):
        return tf.reduce_mean(tf.square(data - label))

    def human(self, data, label):
        return tf.reduce_mean(tf.square(data - label))


# class MnistLoss(LossMethod):
#     # loss_dim means that how many task there is
#     def machine(self, feature_list, label_list):
#         local_loss = tf.Variable(tf.zeros([self._loss_dim]))
#         for index, feature in enumerate(feature_list):
#             predict = tf.nn.softmax(feature)
#             local_loss[index] += -tf.reduce_sum(label_list[index]*tf.log(predict))
#
#         return tf.reduce_sum(local_loss, self._w_loss, axis=)
#
#     def human(self, feature, label):
#         correct_prediction = tf.equal(tf.argmax(data, 1), tf.argmax(label, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
#         return accuracy


# class MnistLossex(LossMethod):
#     def machine(self, data, label):
#         # data = tf.reshape(data, shape=[self._loss_dim])
#         # label = tf.reshape(data, shape=[self._loss_dim])
#         # data = tf.nn.l2_normalize(data, dim=0)
#         data = tf.nn.softmax(data)
#         # final_loss = tf.reduce_sum(-tf.log(1 - tf.abs(data - label)) * self._w_loss)
#         final_loss = -tf.reduce_sum(label * tf.log(data))
#         return final_loss
#
#     def human(self, data, label):
#         correct_prediction = tf.equal(tf.argmax(data, 1), tf.argmax(label, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
#         return accuracy


class MnistLoss(LossMethod):
    def machine(self, data, label, epoch):
        # data = tf.reshape(data, shape=[self._loss_dim])
        # label = tf.reshape(data, shape=[self._loss_dim])


        # rand = tf.random_normal([10], mean=0.0, stddev=1.0)
        # rand = tf.random_uniform([200, 10], minval=0.0, maxval=1.0)

        r_tmp = tf.constant([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)

        # r_tmp = label
        # rand = tf.random_shuffle(r_tmp)
        # rand = tf.random_uniform([100, 10], minval=0.0, maxval=1.0)

        tmp = tf.constant(3, dtype=tf.float32)
        lamda = tf.constant(0.2, dtype=tf.float32)
        #
        # a = np.ones([1, 10])
        # # for i in range(epoch%tmp):
        # a[0][epoch%tmp] = 0
        #
        # np.random.shuffle(a[0])
        # b = a.copy()
        #
        # for i in range(99):
        #     np.random.shuffle(b[0])
        #     a = np.append(a, b, axis=0)
        #
        # rand = tf.constant(a, dtype=tf.float32)
        # rand = tf.random_shuffle(rand)

        rand = tf.random_normal([100, 10], mean=0.0, stddev=1.0 + lamda*tf.floor(epoch/tmp))
        # rand = tf.random_normal([100, 10], mean=0.0, stddev=1.0)
        rand = tf.abs(rand)
        rand = tf.minimum(1.0, rand)
        rand = tf.floor(rand)
        rand = tf.random_shuffle(rand)

        data = tf.nn.softmax(data)
        # data = tf.nn.l2_normalize(data, dim=1)

        data = tf.Print(data, [tf.reduce_max(data, axis=1)], 'data:', summarize=10, first_n=10)
        final_loss = -tf.reduce_sum(tf.log(1 - tf.abs(data - label))*rand)
        # final_loss = -tf.reduce_sum(label * tf.log(data))
        return final_loss

    def human(self, data, label):
        correct_prediction = tf.equal(tf.argmax(data, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
        return accuracy

if __name__ == '__main__':
    # loss_w = LossMethod(task_num=5).loss_weight_var()
    # ttu.print_var(loss_w)
    # loss_b = LossMethod(task_num=5).loss_weight_var()
    # ttu.print_var(loss_b)
    # a = tf.constant([[1], [2], [3]])
    # b = tf.constant([[4], [5], [6]])
    # c = tf.Variable([[0], [0], [0]])
    #
    # c += b
    #
    # c +=
    #
    # ttu.print_var(c)
    # arr = np.array(np.linspace(0, 100, 1))

    a = tf.Variable([0.0, 1.0, 100.0])
    a = tf.nn.l2_normalize(a, dim=0)
    ttu.print_var(a)

    # tf.contrib.layers.batch_nor
