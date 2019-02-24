from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


# import the data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_predition = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))
    # argmax(, axis)Describes which axis of the input Tensor to reduce across.

    accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):

    # truncated_normal：产生随机变量
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)
    # >>>一般bias初始值为正数比较好

    return tf.Variable(initial)


def conv2d(x, W):

    # 使用tf自带的函数创建conv层
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#     函数形式：
#
#     tensorflow.nn.conv2d(
#         input,
#         filter,
#         strides,
#         padding,
#         use_cudnn_on_gpu=True,
#         data_format='NHWC',
#         dilations=[1, 1, 1, 1],
#         name=None
#     )
#
# -----------------------------------------------------------------------------------------
# 用的最多的参数是前四个：
#
# input：用于做卷积运算的数据，格式为一个tensor，
# 拥有四个维度：[batch, in_height, in_width, in_channels]。下面以最常用的图像卷积为例即使四个参数的意义。
#
# batch：数据量，如果对100张图片进行计算，则batch就是100.
#
# in_height：图像的高度方向上的像素
#
# in_width：图像的宽度方向上的像素
#
# in_channels：如果图片是彩色的，那么每个像素点都有rgb三个通道的数据，in_channels为3。
# 如果是黑白图片则in_channels为1。
#
# filter：卷积核信息，同样是一个tensor，四个维度[filter_height, filter_width, in_channels, out_channels]。
#
# filter_height：卷积核的高度
#
# filter_width：卷积核宽度
#
# in_channels：输入数据的通道数，必须与input的in_channels相同
#
# out_channels：卷积核数量。根据某些技术博客所述，一般采用16的倍数，并且随着层级增加增大。out_channel是一个非常自由的参数，也比较重要。我个人理解是，每一个池化层都是将tensor缩并的过程，其作用是将图片分块，提炼出关键信息。这其中必然损失图像或者数据的大量信息，因此为了减少池化过程中的信息损失，增加卷积核数量。增加卷积核可以将图片或者数据信息从不同的角度进行分块、信息提炼。
#
# strides：卷积步长，一个tensor。四个维度分别是[batch_stride, heightstride, widthstride, channel_stride。
# 一般情况下batch_stride和channel_stride都是1，因为一般情况下，很少在图像之间或者通道之间进行卷积运算。
#
# padding：卷积的边缘处理方式，一个string形式的参数，值为“VALID”或者“SAME”。
#
# VALID：卷积核运动到最后，如果右面或者下面的数据 <“卷积核尺寸 + 步长”，则卷积停止。
#
# SAME：如果右面或者下面的数据 > 步长，则继续卷积运算。卷积核超出的部分补0.
#
# -----------------------------------------------------------------------------------------
# conv2d的运算结果为一个四维tensor，大小为[batch, height, width, out_channel]


def max_pool_2x2(x):
    # 用pooling压缩x, y方向尺寸
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    # 第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，
    # 依然是[batch, height, width, channels]这样的shape
    #
    # 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
    # 因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    #
    # 第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride, stride, 1]
    #
    # 第四个参数padding：和卷积类似，可以取'VALID'或者'SAME'
    #
    # 返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(xs, [-1, 28, 28, 1])
# : [n_samples, 28, 28, 1] 1是由于是黑白图像
# : 实质是升维操作


# print(x_image.shape)


### create layers

## conv1 layer

W_conv1 = weight_variable([5, 5, 1, 32])
# : 5，5 代表 patch 5x5 >>>(照相机大小)
# : 1 代表in_size>>>原本feature map 的 width
# : 32 代表out_size>>>提取后feature map 的 width(PS: 长方体变厚了)
# : PS: 32为自行假定

b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# : h_conv1>>> output size nx28x28x32

h_pool1 = max_pool_2x2(h_conv1)
# : h_pool1>>> output size nx14x14x32 缩减尺寸max/mean pooling

tf.summary.histogram('conv_layer1', h_pool1)


## conv2 layer

W_conv2 = weight_variable([5, 5, 32, 64])
# : 5，5 代表 patch 5x5 >>>(照相机大小)
# : 32 代表in_size>>>原本feature map 的 width
# : 64 代表out_size>>>提取后feature map 的 width(PS: 长方体变厚了)

b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# : h_conv1>>> output size nx14x14x64

h_pool2 = max_pool_2x2(h_conv2)
# : h_pool1>>> output size nx7x7x64 缩减尺寸max/mean pooling

tf.summary.histogram('conv_layer2', h_pool2)

## func1 layer : ndim=1

W_func1 = weight_variable([7*7*64, 1024])
b_func1 = weight_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_func1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_func1) + b_func1)
h_func1_drop = tf.nn.dropout(h_func1, keep_prob)

tf.summary.histogram('func_layer1', h_func1_drop)

## func2 layer : ndim=1

W_func2 = weight_variable([1024, 10])
b_func2 = weight_variable([10])

prediction = tf.matmul(h_func1_drop, W_func2) + b_func2

tf.summary.histogram('func_layer2', prediction)

# the loss between prediction amd real data

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=ys))

tf.summary.scalar('loss', cross_entropy)


# train step

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# create a session
sess = tf.Session()

# merge all summary
merged = tf.summary.merge_all()

# create summary writer
train_writer = tf.summary.FileWriter("logs_drop/train", sess.graph)
test_writer = tf.summary.FileWriter("logs_drop/test", sess.graph)

# >>> 这里用到了两个summary

# initialize variables
sess.run(tf.global_variables_initializer())

# start training
for step in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.75})
    if step % 50 == 0:
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
        train_result = sess.run(merged, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: mnist.test.images[:1000],
                                                  ys: mnist.test.labels[:1000], keep_prob: 1})
        train_writer.add_summary(train_result, step)
        test_writer.add_summary(test_result, step)
