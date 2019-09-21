# encoding: utf-8
# Author: zTaylor

import tensorflow as tf
import numpy as np

a = tf.Variable(tf.truncated_normal(shape=[2,2]))

sess = tf.Session()
sess.run(tf.initialize_all_variables())     # !!!!!!!!
print(sess.run(a))

b = a
print(id(a))
print(id(b))

b = tf.expand_dims(b, 0)
print(id(b))
print(b.shape)
print(sess.run(b))

c = tf.expand_dims(a, 1)
print(id(c))
print(c.shape)
print(sess.run(c))

c = tf.expand_dims(a, 2)            # interval [-3, 2]
print(id(c))
print(c.shape)
print(sess.run(c))

d = tf.squeeze(c)
print(d.shape)
print(type(d.shape))
print(sess.run(d))
# wrong:
# print(sess.run(d.shape))              #  Fetch argument TensorShape([Dimension(2), Dimension(2)]) has invalid type <class 'tensorflow.python.framework.tensor_shape.TensorShapeV1'>, must be a string or Tensor. (Can not convert a TensorShapeV1 into a Tensor or Operation.)

print("-----------------------")
# wrong
# print(d.shape())
# print(type(d.shpae()))

print(tf.shape(d))
print(type(tf.shape(d)))
print(sess.run(tf.shape(d)))
print(type(sess.run(tf.shape(d))))

print("++++++++++++++++++++++++")
# tf.shape() can be used for list, np...
print(tf.shape([1,2,3]))
print(type(tf.shape([1,2,3])))
print(sess.run(tf.shape([1,2,3])))
print(sess.run(tf.shape(np.array([1,2,3]))))

print("========================")
print(d.get_shape)
print(d.get_shape())
print(type(d.get_shape))
print(type(d.get_shape()))
print(d.get_shape().as_list())
print(type(d.get_shape().as_list()))

print("++++++++++++++++++++++++++++++++++++++")
print(tf.TensorShape([]))
print(tf.TensorShape([None]))
print(tf.TensorShape([2]))
print(tf.TensorShape([2,2]))
print(type(tf.TensorShape([])))
print(type(tf.TensorShape([2,2])))
a = tf.TensorShape([])
print(a)
print(type(a))