import tensorflow as tf
import numpy as np


x = tf.Variable(0, dtype=tf.float32)

op = tf.assign(x, x+1)

with tf.control_dependencies([op]):
    x = tf.identity(x)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        print(sess.run(x))

# # We define our Variables and placeholders
# x = tf.placeholder(tf.int32, shape=[], name='x')
# y = tf.Variable(2, dtype=tf.int32)
#
# # We set our assign op
# assign_op = tf.assign(y, y + 1)
#
# # We build our multiplication, but this time, inside a control depedency scheme!
# with tf.control_dependencies([assign_op]):
#     # Now, we are under the dependency scope:
#     # All the operations happening here will only happens after
#     # the "assign_op" has been computed first
#     out = x * y
#
# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#
#   for i in range(3):
#     print('output:', sess.run(out, feed_dict={x: 1}))

# def print_num_of_total_parameters(output_detail=False, output_to_logging=False):
#     total_parameters = 0
#     parameters_string = ""
#
#     for variable in tf.trainable_variables():
#
#         shape = variable.get_shape()
#         variable_parameters = 1
#         for dim in shape:
#             variable_parameters *= dim.value
#         total_parameters += variable_parameters
#         if len(shape) == 1:
#             parameters_string += ("%s %d, " % (variable.name, variable_parameters))
#         else:
#             parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))
#
#     if output_to_logging:
#         if output_detail:
#             pass
#             # logging.info(parameters_string)
#         # logging.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
#     else:
#         if output_detail:
#             print(parameters_string)
#         print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
#
# # alias tf_test_utils as ttu
# def print_var(var):
#     with tf.Session() as sess:
#         sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#         val = sess.run(var)
#         return val
#
# def average_mean(var, step):
#     ema = tf.train.ExponentialMovingAverage(0.99, step)  # 初始设定衰减率为0.99
#     print(ema)
#     maintain_averages_op = ema.apply([var])  # 更新列表中的变量
#     get_op = ema.average(var)
#     return maintain_averages_op, get_op
#
# v1 = tf.Variable(5, dtype=tf.float32)  # 定义一个变量，初始值为0
# v2 = tf.Variable(10, dtype=tf.float32)
# step = tf.Variable(1, trainable=False)  # step为迭代轮数变量，控制衰减率
# # B_step = tf.Variable(100, trainable=False)
#
# ret_v1, get_v1 = average_mean(v1, step)
# ret_v2, get_v2 = average_mean(v2, step)
#
# aema = tf.train.ExponentialMovingAverage(0.99, 1)
#
# print(aema)

# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()  # 初始化所有变量
#     sess.run(init_op)
#
#
#
#
#
#     print(sess.run(ret_v1))  # 输出初始化后变量v1的值和v1的滑动平均值
#     sess.run(print_num_of_total_parameters())
#     print(sess.run(get_v1))
#
#
#     print(sess.run(ret_v2))  # 输出初始化后变量v1的值和v1的滑动平均值
#     print(sess.run(get_v2))
    # sess.run(tf.assign(v1, 5))  # 更新v1的值
    # sess.run(maintain_averages_op)  # 更新v1的滑动平均值
    # print(sess.run([v1, ema.average(v1)]))
    #
    # sess.run(tf.assign(step, 1))  # 更新迭代轮转数step
    # sess.run(tf.assign(v1, 10))
    # sess.run(maintain_averages_op)
    # print(sess.run([v1, ema.average(v1)]))
    # 再次更新滑动平均值，
    # sess.run(maintain_averages_op)
    # print(sess.run([v1, ema.average(v1)]))
    # 更新v1的值为15
    # sess.run(tf.assign(v1, 15))
    #
    # sess.run(maintain_averages_op)
    # print(sess.run([v1, ema.average(v1)]))

# tf.nn.batch_normalization()
#
# img = tf.Variable(np.array([[1, 2, 3], [4, 5, 6]]), dtype=tf.float32)
# axis = list(range(len(img.get_shape()) - 1))
# mean, variance = tf.nn.moments(img, [0, 1])
#
# print(print_var([mean, variance]))