from include_pkg import *
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

# alias tf_test_utils as ttu
def print_var(var):
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        val = sess.run(var)
        return val

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def variable_summaries(var):
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
        tf.summary.histogram('his', var)


mnist = input_data.read_data_sets('../data_set/mnist/', one_hot=True)

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

keep_prob = tf.placeholder("float")

tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
B1 = tf.Variable(tf.constant(0.1, tf.float32, [32]))

W_conv2 = weight_variable([3, 3, 32, 64])
B2 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
b_conv2 = bias_variable([64])

W_conv3 = weight_variable([3, 3, 64, 128])
B3 = tf.Variable(tf.constant(0.1, tf.float32, [128]))
b_conv3 = bias_variable([128])

W_fc1 = weight_variable([4 * 4 * 128, 1024])
b_fc1 = bias_variable([1024])

x_image = tf.reshape(x, [-1,28,28,1])

# t1 = conv2d(x_image, W_conv1)
# t1bn, ema1 = batchnorm(t1, tst, iter, B1, convolutional=True)

t1bn = conv2d(x_image, W_conv1) + b_conv1
h_conv1 = tf.nn.relu(t1bn)
h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('l1_out_put'):
    variable_summaries(h_pool1)


# t2 = conv2d(h_pool1, W_conv2)
# t2bn, ema2 = batchnorm(t2, tst, iter, B2, convolutional=True)

t2bn = conv2d(h_pool1, W_conv2)
h_conv2 = tf.nn.relu(t2bn)
h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('l2_out_put'):
    variable_summaries(h_pool2)

# t3 = conv2d(h_pool2, W_conv3)
# t3bn, ema3 = batchnorm(t3, tst, iter, B3, convolutional=True)

t3bn = conv2d(h_pool2, W_conv3)
h_conv3 = tf.nn.relu(t3bn)
h_pool3 = max_pool_2x2(h_conv3)

with tf.name_scope('l1_out_put'):
    variable_summaries(h_pool3)

h_pool2_flat = tf.reshape(h_pool3, [-1, 4*4*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# update_ema = tf.group(ema1, ema2, ema3)

sess = tf.InteractiveSession()

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())

train_writer = tf.summary.FileWriter('./summary/', sess.graph)
merged = tf.summary.merge_all()
for i in range(100):
    batch = mnist.train.next_batch(50)
    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0, tst: False, iter: i})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        summary = sess.run(merged, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0, tst: False, iter: i})
        train_writer.add_summary(summary, i)

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0, tst: False, iter: i})
    # sess.run(update_ema, {x: batch[0], y_: batch[1], keep_prob: 1.0, tst: False, iter: i})

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0,
                                                    tst: False, iter: i}))


# if __name__ == '__main__':
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

    # rand = tf.random_normal([100, 10], mean=0.0, stddev=2.0)
    # rand = tf.abs(rand)
    # rand = tf.minimum(1.0, rand)
    # rand = tf.floor(rand)
    # rand = tf.random_shuffle(rand)
    #
    # print(print_var(rand))

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

    # # print(print_var(tf.nn.softmax(a)))
    #
    # v1 = tf.Variable(0, dtype=tf.float32)  # 定义一个变量，初始值为0
    # step = tf.Variable(0, trainable=False)  # step为迭代轮数变量，控制衰减率
    #
    # ema = tf.train.ExponentialMovingAverage(0.99, step)  # 初始设定衰减率为0.99
    # maintain_averages_op = ema.apply([v1])  # 更新列表中的变量
    #
    # with tf.Session() as sess:
    #     init_op = tf.global_variables_initializer()  # 初始化所有变量
    # sess.run(init_op)
    #
    # print(sess.run([v1, ema.average(v1)]))  # 输出初始化后变量v1的值和v1的滑动平均值
    #
    # sess.run(tf.assign(v1, 5))  # 更新v1的值
    # sess.run(maintain_averages_op)  # 更新v1的滑动平均值
    # print(sess.run([v1, ema.average(v1)]))
    #
    # sess.run(tf.assign(step, 1))  # 更新迭代轮转数step
    # sess.run(tf.assign(v1, 10))
    # sess.run(maintain_averages_op)
    # print(sess.run([v1, ema.average(v1)]))
    # # 再次更新滑动平均值，
    # sess.run(maintain_averages_op)
    # print(sess.run([v1, ema.average(v1)]))
    # # 更新v1的值为15
    # sess.run(tf.assign(v1, 15))
    #
    # sess.run(maintain_averages_op)
    # print(sess.run([v1, ema.average(v1)]))

    # tf.nn.batch_normalization()

    # img = tf.Variable(np.array([[1, 2, 3], [4, 5, 6]]), dtype=tf.float32)
    # axis = list(range(len(img.get_shape()) - 1))
    # mean, variance = tf.nn.moments(img, [0, 1])
    #
    # print(print_var([mean, variance]))
