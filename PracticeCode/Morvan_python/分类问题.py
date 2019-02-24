import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# 函数：增加层方法
def add_layer(inputs,
              in_size,
              out_size,
              n_layer,
              activation_function=None):
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

    layer_name = 'layer_%d' % n_layer

    with tf.name_scope(layer_name):

        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')  # 初始化权重一般是随机变量
            tf.summary.histogram(':', Weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')  # 初始化权重一般是随机变量,且不能为0
            tf.summary.histogram(':', biases)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        tf.summary.histogram(':', outputs)

        return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_predition = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))
    # argmax(, axis)Describes which axis of the input Tensor to reduce across.

    accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# define placeholder for inputs network
xs = tf.placeholder(tf.float32, [None, 784])  # 784 = 28*28 ; [num_sample, ]
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer

prediction = add_layer(xs, 784, 10, 2, activation_function=None)
# as for classificaiton question, we usually use the softmax method as activation_function

# the error between prediction and real data
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=prediction))

train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

# create a session
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})

    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels
        ))






