# encoding: utf-8
# Author: zTaylor

import os
import tensorflow as tf

import config

class SiameseTracker:
    def __init__(self, sess):
        self.sess = sess
        # self.saver = tf.train
        self.build()

    def build(self):
        self.path = tf.train.latest_checkpoint(os.path.dirname(config.model_saving_path))
        self.saver = tf.train.import_meta_graph(self.path + ".meta")
        self.saver.restore(self.sess, self.path)
        self.tensor_list = [item for item in tf.trainable_variables()]
        a = [n.name for n in tf.get_default_graph().as_graph_def().node]
        for item in a:
            print(item)
