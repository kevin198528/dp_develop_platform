import tensorflow as tf
import numpy as np
import os


class DeepLearningFrame(object):

    def __init__(self, DataProcess, NeuralNet, LossMethod):
        self._data_pro = DataProcess
        self._net = NeuralNet
        self._loss = LossMethod

        self._saver = tf.train.Saver()
        self._save_path = self._data_pro.name() + '_ckpt/'
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)

        self._curent_epoch = tf.placeholder(dtype=tf.float32)

        abs_data_output = self._net.forward_pass(self._data_pro.abstract_data())
        self._abs_m_loss = self._loss.machine(abs_data_output, self._data_pro.abstract_label(), self._curent_epoch)
        self._abs_h_loss = self._loss.human(abs_data_output, self._data_pro.abstract_label())

    def train(self, epoch, learn_rat=0.1, show=5):
        # global_step = tf.Variable(0, trainable=False)
        #
        # learning_rate = tf.train.exponential_decay(learn_rat,
        #                                            global_step=global_step,
        #                                            decay_steps=5, decay_rate=0.8)
        #
        #
        # train = tf.train.GradientDescentOptimizer(learning_rate).minimize(self._abs_m_loss)
        #
        # add_global = global_step.assign_add(1)
        #
        train = tf.train.GradientDescentOptimizer(learn_rat).minimize(self._abs_m_loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            merged = tf.summary.merge_all()

            train_writer = tf.summary.FileWriter('./summary/', sess.graph)

            while self._data_pro.latest_epochs < epoch:

                epoch_start = self._data_pro.latest_epochs
                # for each epoch data must be reload one batch
                train_dict = {}
                train_dict.update(self._data_pro.train_dict())
                train_dict.update(self._net.train_dict())
                train_dict.update({self._curent_epoch: self._data_pro.latest_epochs})

                sess.run(train, feed_dict=train_dict)
                epoch_end = self._data_pro.latest_epochs

                # if epoch_start != epoch_end:
                #     sess.run(add_global)
                #     print(sess.run(learning_rate))

                if epoch_start != epoch_end and epoch_start%(epoch/show) == 0:
                    print('epoch:%d' % epoch_start)
                    # for each epoch data must be reload one batch
                    valid_dict = {}
                    valid_dict.update(self._data_pro.valid_dict())
                    valid_dict.update(self._net.valid_dict())
                    valid_dict.update({self._curent_epoch: self._data_pro.latest_epochs})

                    m_valid_loss = sess.run(self._abs_m_loss, feed_dict=valid_dict)
                    h_valid_loss = sess.run(self._abs_h_loss, feed_dict=valid_dict)
                    print('valid_m_loss:%.3f valid_h_loss:%.3f' % (m_valid_loss, h_valid_loss))

                    summary = sess.run(merged, feed_dict=valid_dict)
                    train_writer.add_summary(summary, self._data_pro.latest_epochs)


            # print(sess.run([self.net.get_weights()]))
            check_point_file = self._save_path+self._data_pro.name()+'_'+self._net.name()+'_'\
                               +time.strftime('%Y-%m-%d %H:00:00', time.localtime())
            self._saver.save(sess, check_point_file +'.ckpt', 1)

            train_writer.close()

    def test(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self._save_path)
            if ckpt and ckpt.model_checkpoint_path:
                self._saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('could not find check point path')

            # for each epoch data must be reload one batch
            test_dict = {}
            test_dict.update(self._data_pro.test_dict())
            test_dict.update(self._net.test_dict())
            print('test_h_loss:%.3f' % sess.run(self._abs_h_loss, feed_dict=test_dict))
