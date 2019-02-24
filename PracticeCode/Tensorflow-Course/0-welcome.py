"""
@Author: Atticus Johnson
@Describe: Notes for the Tensorflow-Course 0-welcome
@Modify:2019.1.30
@IDE: pycharm
@python :3.6
@os : win10
"""

# 首先，我们需要导入一些必要的库
from __future__ import print_function
import tensorflow as tf
import os

# 缺省路径是python当前文件夹路径
# 设置'log_dir'变量，缺省值为'./logs'
tf.flags.DEFINE_string(
    'log_dir',
    os.path.dirname(os.path.abspath(__file__)) + '/logs',
    'Directory where event logs are written to'
)

# 存储所有元素在FLAG结构体中
FLAGS = tf.flags.FLAGS

# 我们可以使用FLAGS.调用里面存储的元素，例如：
print(tf.flags.FLAGS.log_dir)

# 在路径存储时，必须使用绝对路径
# 如果不使用绝对路径 用以下语句抛出错误
if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
    raise ValueError('You must assign absolute path for --log_dir')

# 在Tensorflow中 我们可以定义一些句子
welcome = tf.constant('Welcome to Tensorflow world')


# 执行一个会话
# session 是运行操作的环境
# 这里用 with as 可以自动在会话结束后关闭session环境
# tf.summary.FileWritter 用来写事件摘要到event files,
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
    print("output: ", sess.run(welcome))

# 关闭会话
writer.close()
sess.close()


