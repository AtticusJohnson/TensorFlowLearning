from __future__ import print_function
import tensorflow as tf
import os
import numpy as np


tf.flags.DEFINE_string(
    'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
    'Directory where event logs are written to.')

FLAGS = tf.flags.FLAGS


if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
    raise ValueError('You must assign absolute path for --log_dir')
'''----------------------------------------------------------------------------------'''

# tf.constant
# 定义一些常量
a = tf.constant(5, name="a")
b = tf.constant(10, name="b")
tensor_a = 5*tf.ones([5, 5])
tensor_b = 3*tf.ones([5, 5])

# 一些基本的运算
x = tf.add(a, b, name="add")
y = tf.div(a, b, name="divide")

'''----------------------------------------------------------------------------------'''

# Run the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
    print("a =", sess.run(a))
    print("b =", sess.run(b))
    print("a + b =", sess.run(x))
    print("a/b =", sess.run(y))

# Closing the writer.
writer.close()
sess.close()

