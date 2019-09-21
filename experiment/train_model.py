# encoding: utf-8
# Author: zTaylor

import os
import sys
import random
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))    # 不加也可导入
sys.path.append(config.ROOT_PATH)
from model import siam_model_for_test   # 名字不能有横线


# todo: setting seeds is not working
# np.random.seed(config.seed)
# # tf.random.set_seed(config.seed)   # 1.14edition
# tf.random.set_random_seed(config.seed)

def main():
    g = tf.Graph()
    with g.as_default():
        random.seed(config.seed)
        np.random.seed(config.seed)
        tf.random.set_random_seed(config.seed)
        model = siam_model_for_test.SiameseModel()
        # with tf.Session() as sess:
        #     # sess.run(tf.initialize_all_variables())
        #     model.train(sess, g)
        model.train()

if __name__ == "__main__":
    main()
    pass