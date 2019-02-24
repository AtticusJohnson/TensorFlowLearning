import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
# 此处的image矩阵，其值介于0-1之间
# "MNIST_data/"为当前文件夹创建一个名字为"MNIST_data/"的文件夹
# 用来存放数据
# one-hot向量是指在大多数维度上数值为0，仅在其中一个维度上数值为1的向量。
# 在这种情况下，第n个数字将被表示为在第n维中为1的向量。

# MNIST数据被分为三部分：55,000个训练数据（mnist.train）
# 10,000个测试数据（mnist.test）和5,000个验证数据（mnist.validation）
# 这种切分是非常重要的：它能通过一部分我们并没有实际用来训练学习的数据
# 来确保我们的算法有很好的通用性。

# 如前所述，每个MNIST数据点有两个部分：手写数字的图像和相应的标签
# 我们称为图像”x”和标签”y”。训练集和测试集都包含图像及其相应的标签
# 例如训练图像是mnist.train.images 训练标签是mnist.train.labels。

# 这是一个softmax回归的典型案例。如果你想给一个对象赋予其表示不同数字的概率
# 可以使用softmax，因为softmax可以得出一组介于0到1之间的值
# 并且这组值加起来结果为1。即使在以后，当我们训练其他更复杂的模型时
# 最后一步也是一层softmax


def mnist_model(features, labels, net_structure):

    w = tf.get_variable('w', net_structure, dtype=tf.float32)
    b = tf.get_variable('b', net_structure[1], dtype=tf.float32)

    y_in = tf.nn.softmax(tf.matmul(features, w) + b)

    loss_in = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                         logits=y_in)

    return loss_in, y_in

'''
    global_step = tf.train.get_global_step()
    optimizer = tf.train.AdamOptimizer(0.1)
    train = tf.group(tf.assign.add(global_step, 1), optimizer.minimize(loss))

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=y,
                                      loss=loss,
                                      train_op=train)


estimator = tf.estimator.Estimator(model_fn=mnist_model)

'''
mnist_params_input = mnist.train.images.shape[1]

try:
    mnist_params_output = mnist.train.labels.shape[1]
except IndexError:
    mnist_params_output = 1

structure = [mnist_params_input, mnist_params_output]

x = tf.placeholder(tf.float32, [None, mnist_params_input])

y_ = tf.placeholder(tf.float32, [None, mnist_params_output])


loss, y = mnist_model(x, y_, structure)

train_step = tf.train.AdamOptimizer(0.1).minimize(loss)


first_sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
for _ in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)

    first_sess.run(train_step, feed_dict={x: batch_x, y_: (batch_y/9)[:, np.newaxis]})

try:
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
except ValueError:
    correct_prediction = tf.equal(y, y_)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(first_sess.run(accuracy, feed_dict={x: mnist.test.images,
                                          y_: (mnist.test.labels/9)[:, np.newaxis]}))

