# encoding: utf-8
# Author: zTaylor

"""Build SiamFC model for color image series"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import config
import scripts
from data_processor import batch_loader

class SiameseModel:
    def __init__(self):
        with tf.variable_scope("initial"):    # Graph will be specified in training py_file
            # to#do 不需要把z的改batch_size改为8！!!!!!!!!!!!!!!!!!!!   反而不利于计算feature_map        改回1！！！！！！！！！！
            # self.z = tf.placeholder(dtype=tf.float32, shape=config.z_batch, name="input_z")    # z only is 1 image
            # self.x = tf.placeholder(dtype=tf.float32, shape=config.x_batch, name="input_x")
            # to#do y_true?????
            # self.y_true = tf.placeholder(dtype=tf.float32, shape=config.y_true_batch, name="label_y_true")    # omit channel, since it is 1

            self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")    # indicating whether network used for training or not
            # self.model_saver = tf.train.Saver(max_to_keep=10, name="Model Saver")

            self.training_batch = batch_loader.batch_loader(True)
            # self.validation_batch = batch_loader.batch_loader(False)      # todo
            self.z = self.training_batch.exemplar_batch
            self.x = self.training_batch.search_batch

        self.build_network()


    def build_network(self):
        self.build_y_true(config.batch_size, config.w_num, 8, 16)     # todo: defining stride and R in config_file
        self.build_infer()
        self.build_loss()


    def build_infer(self):
        """
        # build network backbone
        build convolution part
        infer() <-- conv() <-- layer()
        :return: none
        """
        with tf.variable_scope("infer"):
            # to#do how to calculate z_con_out just once??
            # in the inference phase, z need calculate every time, since the nerwork is changed during every iteration
            # in fact, in testing phase, it does need to reduce z's calculation times
            z_conv_out = self.build_conv(self.z, "exemplar_branch")    # shape (1, 6, 6, 128) -> (6, 6, 128, 1)
            x_conv_out = self.build_conv(self.x, "search_branch")      # shape (8 ,22, 22, 128)
            self.cross_corr = tf.nn.conv2d(x_conv_out, z_conv_out, strides=[1,1,1,1], padding="VALID", name="cross_correlation")    # shape (8, 17, 17, 1)      # "cross correlation" is not legal


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
        kernel = tf.Variable(tf.truncated_normal(shape=[para["kernel_size"], para["kernel_size"], para["in_channel"], para["out_channel"]],
                                                 stddev=0.1, name="kernel_initial"),
                             trainable=True, name="kernel")
        conv = tf.nn.conv2d(input, filter=kernel, strides = para["stride"], padding=para["is_padding"], name="conv_out")
        if(para["is_batch_normal"]):
            conv = tf.layers.batch_normalization(conv, training=self.is_training, name="batch_normalization")
        if(para["is_ReLU"]):
            conv = tf.nn.relu(conv, name="ReLU")
        if(para["is_pooling"]):
            conv = tf.nn.max_pool(conv, ksize=para["pooling_size"], strides=para['pooling_stride'], padding=para["is_padding"], name="max_pool")

        return conv


    def build_loss(self):
        """
        build logistic loss

        :return: none
        """
        with tf.variable_scope("loss"):
            # to#do z(exemplar)'s batch_size is only 1
            # todo not use tf.map_fn()??
            # self.loss = tf.map_fn(fn=lambda corr, y_true : self.logistic_loss(corr, y_true),
            #                       elems=(self.cross_corr, self.y_true), name="loss")

            self.loss = self.logistic_loss(self.cross_corr, self.y_true)


    def logistic_loss(self, corr, y_true):
        # # corr shape: (8, 17, 17, 1)    # todo y_true shape: (1, 17, 17, 1)
        # squeezed_corr = tf.squeeze(corr, name="squeezed corr")                             # shape (8, 17, 17)
        # squeezed_y_true = tf.expand_dims(tf.squeeze(y_true, name="squeezed y_true"), 0)    # shape (17, 17) -> (1, 17, 17)        # todo expand_dims' name
        # tiled_y_true = tf.tile(squeezed_y_true, [8])                                       # shape (8, 17, 17)
        # return tf.reduce_sum(tf.log(tf.multiply(squeezed_corr, tiled_y_true)))             # todo ÷ (8 * 17 * 17)         # todo tf.reduce_sum()和tf.log(), 直接计算了整个batch

        # corr shape: (8, 17, 17, 1)
        # y_true shape: (8, 17, 17)
        squeezed_corr = tf.squeeze(corr, name="squeezed_corr")      # shape (8, 17, 17)     # 'squeezed corr' is not a valid scope name
        return tf.reduce_sum(tf.log(1 + tf.exp(tf.negative(tf.multiply(squeezed_corr, y_true))))) / (config.batch_size * config.w_num * config.w_num)
        # Input 'y' of 'Mul' Op has type int32 that does not match type float32 of argument 'x'.
        # 一开始，忘记加tf.exp(tf.negative())，但加了后loss保持不变（无论是0.001、0.01、batch_norm）   (0.6931472)


    def build_y_true(self, batch_size, map_size, stride, R):
        # sc_map = tf.ones(shape=[map_size, map_size], name="initial map")
        # np_map = np.ones(shape=[map_size, map_size], dtype=np.int)        # it will repeat batch_size times later
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
        with tf.variable_scope("build_y_true"):     # "build y_true" is not legal, why?
            self.y_true = tf.constant(value=np_map, dtype=tf.float32, name="label_y_true")


    def train(self, sess):
        """
        train model

        :return:
        """
        self.model_saver = tf.train.Saver(max_to_keep=10, name="Model_Saver")
        with tf.variable_scope("training"):
            # todo
            self.train_step = tf.train.GradientDescentOptimizer(config.lr).minimize(self.loss)  # todo 在config中定义优化器?
            sess.run(self.training_batch.iterator.initializer)                # todo

            for i in tqdm(range(1, config.training_steps)):
                # print(i)

                # z_batch, x_batch = scripts.get_batch(config.batch_size)     # todo
                # y_true_batch = scripts.get_y_true(config.batch_size)        # todo
                # # y_true = ...

                # z_batch, x_batch = self.training_batch.get_one_batch(sess)
                # no need to call get_one_batch() every time

                # y_true_batch = tf.map_fn()      # to#do !!!!!!!!!!!!
                _, loss = sess.run([self.train_step, self.loss],
                                   feed_dict={
                                              # self.z: self.training_batch.exemplar_batch, self.x: self.training_batch.search_batch,
                                              # self.y_true: y_true_batch,
                                              self.is_training: tf.Variable(initial_value=True, trainable=False)})
                if(i % 100 == 0):
                    _, loss, t = sess.run([self.train_step, self.loss, self.cross_corr],
                                       feed_dict={
                                           # self.z: self.training_batch.exemplar_batch, self.x: self.training_batch.search_batch,
                                           # self.y_true: y_true_batch,
                                           self.is_training: tf.Variable(initial_value=True, trainable=False)})
                    print(loss)
                    print(t)
                # if(i % config.summary_interval):
                if(i % config.model_saving_interval == 0):
                    self.model_saver.save(sess, save_path=config.model_saving_path, global_step=i)


    def test(self):
        with tf.variable_scope("test"):
            # todo 不能和training phase用相同的build函数！ testing时x也是只有一张图
            pass
