from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.3)


# 函数：增加层方法
def add_layer(inputs,
              in_size,
              out_size,
              n_layer,
              activation_function=None,
              keep_prob=1):
    """
    function: 增加层方法
    Parameters:
        inputs 输入Tensor或者np.array
        in_size 输入维度 例如 输入张量为[10000, 784], in_size则为784
        out_size 输出维度，即神经元个数 或标签层维度
        activation_function 激活函数 例如 sigmoid relu 等
    Returns:
        outputs 输出Tensor
    """

    layer_name = 'layer_%s' % n_layer

    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')  # 初始化权重一般是随机变量
    tf.summary.histogram(layer_name + '/Weights', Weights)

    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')  # 初始化权重一般是随机变量,且不能为0
    tf.summary.histogram(layer_name + '/biases', biases)

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # Dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    tf.summary.histogram(layer_name + '/outputs', outputs)

    return outputs


# 函数：获得batch

def next_batch(train_data, train_target, batch_size):
    index = [i for i in range(0, train_target.shape[0])]
    np.random.shuffle(index)
    batch_data = np.zeros([batch_size, train_data.shape[1]], dtype=np.float32)
    batch_target = np.zeros([batch_size, train_target.shape[1]], dtype=np.float32)
    for i in range(0, batch_size):
        batch_data[i, :] = train_data[index[i], :]
        batch_target[i, :] = train_target[index[i], :]
    if np.ndim(batch_data) == 1:
        batch_data = batch_data[:, np.newaxis]
    if np.ndim(batch_target) == 1:
        batch_target = batch_target[:, np.newaxis]

    return batch_data.astype(np.float32), batch_target.astype(np.float32)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_predition = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))
    # argmax(, axis)Describes which axis of the input Tensor to reduce across.

    accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8*8
ys = tf.placeholder(tf.float32, [None, 10])


# add output layer
layer_1 = add_layer(xs, 64, 50, '1',
                    activation_function=tf.nn.tanh, keep_prob=keep_prob)
# >>>此处设置50个神经元，是为了故意使网络过拟合
# >>>此处用tanh方法，避免梯度下降时出现数据中出现None的情况
prediction = add_layer(layer_1, 50, 10, '2',
                       activation_function=None, keep_prob=keep_prob)


# the loss between prediction amd real data
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=ys))
tf.summary.scalar('loss', cross_entropy)


# train step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# create a session
sess = tf.Session()

# merge all summary
merged = tf.summary.merge_all()

# create summary writer
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)
# >>> 这里用到了两个summary

# initialize variables
sess.run(tf.global_variables_initializer())

# start training
for step in range(1000):
    batch_x, batch_y = next_batch(X_train, y_train, 100)
    sess.run(train_step, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 1})
    if step % 50 == 0:
        print(compute_accuracy(X_test, y_test))
        train_result = sess.run(merged, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, step)
        test_writer.add_summary(test_result, step)


