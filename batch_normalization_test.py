import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tf_test_utils as ttu


x = tf.Variable(0.0)
print(x)
x_plus_1 = tf.assign_add(x, 1)

tf.identity()

with tf.control_dependencies([x_plus_1]):
    y = x
    print(y)
    #z=tf.identity(x,name='x')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(5):
        print(sess.run(y))


