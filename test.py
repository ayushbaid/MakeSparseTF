# Test MakeSparse operation

import tensorflow as tf
from make_sparse_op import make_sparse


x = tf.Variable(tf.random_normal([3, 3], dtype=tf.float32))
k = tf.constant(2, dtype=tf.int32, shape=[])

init = tf.global_variables_initializer()

with tf.Session():
    init.run()
    print(x.eval())
    print(make_sparse(x, k).eval())
