# Test MakeSparse operation

import tensorflow as tf
from make_sparse_op import make_sparse, make_sparse_grad


x = tf.Variable(tf.random_normal([3, 3], dtype=tf.float32))
k = tf.constant(2, dtype=tf.int32, shape=[])
x_sparse = make_sparse(x, k)

dy = tf.random_normal([3,3], dtype=tf.float32)
dx = make_sparse_grad(x, dy, k)


init = tf.global_variables_initializer()

with tf.Session():
    init.run()
    print(x.eval())
    print(x_sparse.eval())
    print(dx.eval())
