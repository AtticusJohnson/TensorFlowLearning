import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28  # MNIST data input (img shape: 28*28)
n_step = 28  # time steps
n_hidden_units = 256  # neuron in hidden layer
n_classes = 10  # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_step, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.truncated_normal(
        [n_inputs, n_hidden_units], stddev=0.1)),
    # (128, 10)
    'out': tf.Variable(tf.truncated_normal(
        [n_hidden_units, n_classes], stddev=0.1))}

biases = {
    # (128, )
    'in': tf.Variable(tf.constant(
        0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(
        0.1, shape=[n_classes, ]))}


def RNN(X, weights, biases):

    # hidden layer for input to cell
    ############################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X ==> (128 batch * 28 steps, 128 hiddens)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X ==> (128 batch, 28 steps, 128 hiddens)
    X_in = tf.reshape(X_in, [-1, n_step, n_hidden_units])

    # cell
    ############################################
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
        n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # : forget_bias 是Gate的偏置，一开始的偏置建议为1.0，
    # : 当weights = 0, bias = 1 , 0 * inputs + 1 = 1

    # lstm cell is divided into two parts(c_state, m_state)
    # : c_state 是主线；m_state 是分线
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in,
                                        initial_state=_init_state,
                                        time_major=False)
    # : RNN每一个cell计算的结果是一个state
    # : dynamic rnn 是一个比较好的形式，一般就用这个
    # ： time major > 如果 steps：28 在X_in 的第一维度， 则为True; 若不在第一维度，则为False


    # hidden layer for outputs as the final results

    # Method 1
    # results = tf.matmul(states[1], weights['out'] + biases['out'])
    # : states[1]分线剧情结果

    # Method 2 for all conditons
    # unstack to list [(batch, outputs)...]* steps
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # : transpose 矩阵转置 [n_step, batch_size, output_size]
    # unstack 将Tensor转换为list

    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    # : outputs[-1]最后一个output，作为输出与图片标签相对应
    # : 实际上，在本次模型中，由于将最后一个cell的输出作为输出，output[-1] == state[1]

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_step, n_inputs])
        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs, y: batch_ys
            }))
        step += 1
