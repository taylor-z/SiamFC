# encoding: utf-8
# Author: zTaylor

import numpy as np
import tensorflow as tf

import config
from evaluation.archived_siam_tracker import SiameseTracker

def main():
    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            tracker = SiameseTracker(sess)



if __name__ == "__main__":
    main()