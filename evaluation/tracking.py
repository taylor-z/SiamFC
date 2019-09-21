# encoding: utf-8
# Author: zTaylor

import os
import tensorflow as tf

from model.siam_model_for_test import SiameseModel
import config

def main():
    # restore_path = os.path.dirname(config.model_saving_path)
    g = tf.Graph()
    with g.as_default():
        trakcer = SiameseModel(is_training=False)
        with tf.Session() as sess:
            # sess.run(tf.initialize_all_variables())
            trakcer.tracking(sess, g)


if __name__ == "__main__":
    main()