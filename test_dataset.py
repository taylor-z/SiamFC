# encoding: utf-8
# Author: zTaylor

import tensorflow as tf
import itertools
tf.enable_eager_execution()

def gen():
  for i in itertools.count(1):
    yield (i, [1] * i)

ds = tf.data.Dataset.from_generator(
    gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))

for value in ds.take(2):
  print(value)