# encoding: utf-8
# Author: zTaylor

"""Build SiamFC model for color image series"""

import re
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import config
import scripts
from data_processor import batch_loader
from scripts import training_data


class SiameseModel:
    def __init__(self, is_training=True):
        with tf.variable_scope("initial"):    # Graph will be specified in training py_file
            self.is_training = is_training
            self.is_training_tensor = tf.placeholder(shape=[], dtype=tf.bool, name="is_training_tensor")    # indicating whether network used for training or not
            # self.model_saver = tf.train.Saver(max_to_keep=10, name="Model Saver")

            if self.is_training:
                self.training_batch = batch_loader.batch_loader(True)
                self.z = self.training_batch.exemplar_batch
                self.x = self.training_batch.search_batch
                # self.validation_batch = batch_loader.batch_loader(False)      # todo

                # self.global_step = tf.Variable(0, trainable=False, name="global_step")
                # self.lr = tf.train.exponential_decay(config.lr, self.global_step,
                #                                      250, 0.99, name="decay_lr")

            if not self.is_training:
                self.test_batch = batch_loader.batch_loader(is_training=False)
                self.z = self.test_batch.exemplar_batch
                self.x = self.test_batch.search_batch

        self.build_network()


    def build_network(self):
        if self.is_training:
            self.build_y_true(config.batch_size, config.w_num, 8, 16)     # todo: defining stride and R in config_file
            self.build_infer()
            self.build_loss()
        else:
            self.build_infer()


    def build_infer(self):
        """
        # build network backbone
        build convolution part
        infer() <-- conv() <-- layer()
        :return: none
        """
        with tf.variable_scope("infer"):
            # todo how to calculate z_con_out just once??
            # in the inference phase, z need calculate every time, since the nerwork is changed during every iteration
            # in fact, in testing phase, it does need to reduce z's calculation times
            self.normed_z = tf.layers.batch_normalization(self.z, training=self.is_training_tensor)
            self.normed_x = tf.layers.batch_normalization(self.x, training=self.is_training_tensor)
            self.z_conv_out = self.build_conv(self.normed_z, "exemplar_branch", reuse=False)    # shape (1, 6, 6, 128) -> (6, 6, 128, 1)
            self.x_conv_out = self.build_conv(self.normed_x, "search_branch", reuse=True)      # shape (8 ,22, 22, 128)
            self.cross_corr = tf.nn.conv2d(self.x_conv_out, self.z_conv_out, strides=[1,1,1,1], padding="VALID", name="cross_correlation")    # shape (8, 17, 17, 1)      # "cross correlation" is not legal
            self.cross_corr = tf.layers.batch_normalization(self.cross_corr, training=self.is_training, name="BN_for_Cross_Corr")
            self.cross_corr = tf.squeeze(self.cross_corr, name="squeezed_corr")


    def build_conv(self, input, branch, reuse=tf.AUTO_REUSE):
        # todo network backbone
        # with tf.variable_scope(branch):
        with tf.variable_scope("conv", reuse=reuse):    # reuse需要同一scope！！！！？？？？
            logits = input                                                  # todo 值传递？引用？会在“infer”内创建新的tensor么？
            for i, para in enumerate(config.conv_arch):
                with tf.variable_scope("conv_layer_{}".format(i)):
                    logits = self.build_layer(logits, para)                 # todo 每个layer都会有logits么？
            conv_out = logits                                               # todo 是否会出错？(for外使用i会出错?)
            if(branch ==  "exemplar_branch"):
                conv_out = tf.expand_dims(tf.squeeze(conv_out), -1)         # (1, 6, 6, 128) -> (6, 6, 128, 1)   # todo name都要写？ 在tensorboard中出现几个tensor?
        return conv_out


    def build_layer(self, input, para):
        """
        basing gave parameter to create layer

        :param input:
        :param para:
        :return: one layer output
        """
        # todo use tf.layers.conv2d or keras.layers.Conv2D
        # kernel = tf.Variable(tf.truncated_normal(shape=[para["kernel_size"], para["kernel_size"], para["in_channel"], para["out_channel"]],
        #                                          stddev=0.1, name="kernel_initial"),
        #                      trainable=True, name="kernel")
        kernel = tf.get_variable(name="kernel", shape=[para["kernel_size"], para["kernel_size"], para["in_channel"], para["out_channel"]],
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input, filter=kernel, strides = para["stride"], padding=para["is_padding"], name="conv_out")
        if(para["is_batch_normal"]):
            conv = tf.layers.batch_normalization(conv,training=self.is_training_tensor, name="batch_normalization")
        if(para["is_ReLU"]):
            conv = tf.nn.relu(conv, name="ReLU")        # todo defining activator in config
        if(para["is_pooling"]):
            conv = tf.nn.max_pool(conv, ksize=para["pooling_size"], strides=para['pooling_stride'], padding=para["is_padding"], name="max_pool")

        return conv


    def build_loss(self):
        """
        build logistic loss

        :return: none
        """
        with tf.variable_scope("loss"):
            self.loss = self.logistic_loss(self.cross_corr, self.y_true)
            tf.summary.scalar("loss", self.loss)


    def logistic_loss(self, corr, y_true):
        """

        :param corr:
        :param y_true:
        :return:
        """
        # corr shape: (8, 17, 17, 1)
        # y_true shape: (8, 17, 17)
        # squeezed_corr = tf.squeeze(corr, name="squeezed_corr")      # shape (8, 17, 17)     # 'squeezed corr' is not a valid scope name
        return tf.reduce_sum(tf.log(1 + tf.exp(tf.negative(tf.multiply(corr, y_true))))) / (config.batch_size * config.w_num * config.w_num)
        # return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=squeezed_corr, name="sigmoid_loss"))


    def build_y_true(self, batch_size, map_size, stride, R):
        np_map = np.empty(shape=[map_size, map_size], dtype=np.int)
        center_idx = (int(map_size / 2), int(map_size / 2))
        # todo not use "for"?
        for i in range(map_size):
            for j in range(map_size):
                dist = np.sqrt(np.square(i-center_idx[0]) + np.square(j-center_idx[1]))
                if(dist < R / stride):
                    np_map[i][j] = 1
                else:
                    np_map[i][j] = -1

        np_map = np.tile(np_map, [batch_size, 1, 1])
        with tf.variable_scope("build_y_true"):
            self.y_true = tf.constant(value=np_map, dtype=tf.float32, name="label_y_true")


    # def train(self, sess, g):
    #     """
    #     train model
    #
    #     :return:
    #     """
    #     model_saver = tf.train.Saver(max_to_keep=config.max_saving_nums, name="Model_Saver")
    #     restore_path = tf.train.latest_checkpoint(config.restore_path)
    #
    #     sess.run(self.training_batch.iterator.initializer)
    #     if restore_path:
    #         model_saver.restore(sess, restore_path)
    #         trained_step = int(re.findall(r"\d+", restore_path)[-1])
    #         global_step = tf.Variable(trained_step + 1, trainable=False, name="global_step")
    #         for i in tqdm(range(1, trained_step+1)):
    #             _ = sess.run([self.z, self.x])
    #     else:
    #         trained_step = 0
    #         global_step = tf.Variable(1, trainable=False, name="global_step")
    #
    #     with tf.variable_scope("training"):
    #         lr = tf.train.exponential_decay(config.lr, global_step, 250, 0.99, name="decay_lr")
    #         tf.summary.scalar("leatning_rate", lr)
    #
    #         train_step = tf.train.GradientDescentOptimizer(lr).minimize(self.loss, global_step=global_step)  # todo 在config中定义优化器?
    #         is_training_isntance = tf.Variable(initial_value=True, trainable=False, name="is_training_instance")
    #
    #         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #         train_op = tf.group([train_step, update_ops])
    #
    #     with tf.variable_scope("summary"):
    #         merged_summary_op = tf.summary.merge_all()      # 一开始忘记() 在tensorboard中没有scalar       问题：不加()，在sess.run中进行的会是什么？？
    #         summary_writer = tf.summary.FileWriter(config.parameters_saving_path, g)
    #
    #     sess.run(tf.initialize_all_variables())
    #     g.finalize()
    #
    #     for i in tqdm(range(trained_step+1, config.training_steps+1)):
    #         try:
    #             if(i % config.show_interval == 0):
    #                 _lr, cc, loss, z, x = sess.run([lr, self.cross_corr, self.loss, self.z_conv_out, self.x_conv_out],
    #                                               feed_dict={self.is_training_tensor: is_training_isntance})
    #                 print("\nloss: ", loss)
    #                 print("learning rate: ", _lr)
    #
    #                 if(i % config.summary_interval):
    #                     summary_str = sess.run(merged_summary_op)
    #                     summary_writer.add_summary(summary_str, i)
    #
    #                 if (i % config.model_saving_interval == 0):         # assuming / limiting the saving and summary interval is multiple of show_interval
    #                     model_saver.save(sess, save_path=config.model_saving_path, global_step=i)
    #
    #             _ = sess.run(train_op,
    #                          feed_dict={self.is_training_tensor: is_training_isntance})
    #
    #         except tf.errors.OutOfRangeError:
    #             sess.run(self.training_batch.iterator.initializer)


    def train(self):
        """
        train model

        :return:
        """
        model_saver = tf.train.Saver(max_to_keep=config.max_saving_nums, name="Model_Saver")
        restore_path = tf.train.latest_checkpoint(config.restore_path)

        sess = tf.Session()
        sess.run(self.training_batch.iterator.initializer)
        if restore_path:
            model_saver.restore(sess, restore_path)
            trained_step = int(re.findall(r"\d+", restore_path)[-1])
            global_step = tf.Variable(trained_step + 1, trainable=False, name="global_step")
            for i in tqdm(range(1, trained_step+1)):
                _ = sess.run([self.z, self.x])
        else:
            trained_step = 0
            global_step = tf.Variable(1, trainable=False, name="global_step")

        with tf.variable_scope("training"):
            lr = tf.train.exponential_decay(config.lr, global_step, 250, 0.99, name="decay_lr")
            tf.summary.scalar("leatning_rate", lr)

            train_step = tf.train.GradientDescentOptimizer(lr).minimize(self.loss, global_step=global_step)  # todo 在config中定义优化器?
            is_training_isntance = tf.Variable(initial_value=True, trainable=False, name="is_training_instance")

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group([train_step, update_ops])

        with tf.variable_scope("summary"):
            merged_summary_op = tf.summary.merge_all()      # 一开始忘记() 在tensorboard中没有scalar       问题：不加()，在sess.run中进行的会是什么？？
            summary_writer = tf.summary.FileWriter(config.parameters_saving_path, tf.get_default_graph())

        sess.run(tf.initialize_all_variables())
        tf.get_default_graph().finalize()

        for i in tqdm(range(trained_step+1, config.training_steps+1)):
            try:
                if(i % config.show_interval == 0):
                    _lr, cc, loss, z, x = sess.run([lr, self.cross_corr, self.loss, self.z_conv_out, self.x_conv_out],
                                                  feed_dict={self.is_training_tensor: is_training_isntance})
                    print("\nloss: ", loss)
                    print("learning rate: ", _lr)

                    if(i % config.summary_interval):
                        summary_str = sess.run(merged_summary_op)
                        summary_writer.add_summary(summary_str, i)

                    if (i % config.model_saving_interval == 0):         # assuming / limiting the saving and summary interval is multiple of show_interval
                        model_saver.save(sess, save_path=config.model_saving_path, global_step=i)

                _ = sess.run(train_op,
                             feed_dict={self.is_training_tensor: is_training_isntance})

            except tf.errors.OutOfRangeError:
                sess.run(self.training_batch.iterator.initializer)
        sess.close()


    def test(self):
        with tf.variable_scope("test"):
            # todo 不能和training phase用相同的build函数！ testing时x也是只有一张图
            pass


    def tracking(self, sess, g):
        model_saver = tf.train.Saver(max_to_keep=config.max_saving_nums, name="Model_Saver")
        restore_path = tf.train.latest_checkpoint(config.restore_path)

        if restore_path:
            model_saver.restore(sess, restore_path)
        else:
            return None

        is_training_instance = tf.Variable(initial_value=False, trainable=False, name="is_training_instance")
        sess.run(self.test_batch.iterator.initializer)
        sess.run(tf.variables_initializer([is_training_instance]))  # todo 不initialize也能继续运行，为什么没有initialize时不可sess.run(is_training_instance)???
        g.finalize()

        for i in tqdm(range(10000)):
            try:
                print(self.cross_corr)
                print(sess.run(self.cross_corr, feed_dict={self.is_training_tensor : is_training_instance})[:, 6:11, 6:11])
                break
            except tf.errors.OutOfRangeError:
                break