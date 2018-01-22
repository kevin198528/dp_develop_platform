from include_pkg import *
import tensorflow as tf
import numpy as np

# alias tf_test_utils as ttu
def print_var(var):
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        val = sess.run(var)
        return val
        # print('type : ', end='')
        # print(type(val))
        # print('value : ', end='')
        # print(val)




if __name__ == '__main__':
    # data = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    # var = tf.Variable(data, dtype=tf.float32)
    # data = tf.linspace(1.0, 10.0, 5)



    # n_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # data = tf.constant(n_data, dtype=tf.float32)
    # # print(n_data.shape)
    #
    # # ret = tf.layers.batch_normalization(data, axis=0)
    # ret = tf.reduce_max(data, axis=1)
    #
    # # ret = tf.random_uniform([10], minval=0.0, maxval=1.0)
    #
    # a = tf.constant([1, 0, 0, 0, 0,], dtype=tf.float32)
    # ret = tf.random_shuffle(a)

    # a = tf.ones([2, 5], dtype=tf.float32)
    # b = tf.zeros([2, 5], dtype=tf.float32)
    #
    # c = tf.concat([a, b], 1)

    # c = tf.random_normal([10, 10], mean=0.0, stddev=10.0)

    # c =tf.random_uniform([5, 5], minval=0.0, maxval=1.0)

    # c =tf.truncated_normal([100], mean=0.0, stddev=0.5)

    # ret = tf.nn.l2_normalize(data, dim=[1])

    # ret = tf.random_shuffle(c)

    # ret = tf.abs(c)
    # ret = tf.minimum(ret, 1)
    # ret = tf.floor(ret)



    # a = tf.ones([9])
    # b = tf.zeros([1])
    # c = tf.concat([a, b], axis=0)
    # d = tf.random_shuffle(c)
    #
    # for i in range(8):
    #     t1 = tf.ones([9])
    #     t2 = tf.zeros([1])
    #     t3 = tf.concat([t1, t2], axis=0)
    #     t4 = tf.random_shuffle(t3)
    #     d = tf.concat([d, t4], axis=0)
    #
    #
    # print_var(d)



    # a = np.array([[0, 1, 1, 1, 1, 1, 1]])

    # a = np.array([[0, 1, 1, 1, 1, 1, 1]])
    # b = a.copy()
    # for i in range(9):
    #     np.random.shuffle(b[0])
    #     a = np.append(a, b, axis=0)
    #
    # print(a)

    # a = np.ones([1, 10])
    # a[0][0] = 0
    # np.random.shuffle(a[0])
    # b = a.copy()
    #
    # for i in range(9):
    #     np.random.shuffle(b[0])
    #     a = np.append(a, b, axis=0)
    #
    # print(a)

    rand = tf.random_normal([100, 10], mean=0.0, stddev=2.5)
    rand = tf.abs(rand)
    rand = tf.minimum(1.0, rand)
    rand = tf.floor(rand)
    rand = tf.random_shuffle(rand)

    print(print_var(rand))

    # a = tf.ones([10, 7, 7, 32])
    #
    # filter = tf.constant(shape=[7, 7, 32, 10], value=0.25)
    #
    # b = tf.nn.conv2d(a, filter, strides=[1, 1, 1, 1], padding='SAME')
    #
    # c = print_var(b)
    # print(c.shape)

    # global_step = tf.Variable(0, trainable=False)
    #
    # initial_learning_rate = 0.1  # 初始学习率
    #
    # learning_rate = tf.train.exponential_decay(initial_learning_rate,
    #                                            global_step=global_step,
    #                                            decay_steps=10, decay_rate=0.9)
    # opt = tf.train.GradientDescentOptimizer(learning_rate)
    #
    # add_global = global_step.assign_add(1)
    # with tf.Session() as sess:
    #     tf.global_variables_initializer().run()
    #     print(sess.run(learning_rate))
    #     for i in range(10):
    #         _, rate = sess.run([add_global, learning_rate])
    #         print(rate)


    # a = tf.constant([[1 , 2, 5], [2, 3, 4]], dtype=tf.float32)

    # a = tf.constant(11, dtype=tf.int32)
    # b = tf.constant(3, dtype=tf.int32)
    #
    # print(print_var(a%b))

    # print(print_var(tf.nn.softmax(a)))

