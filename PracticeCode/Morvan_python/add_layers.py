import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 构造训练数据

x_data = np.linspace(-1, 1, 300)[:, np.newaxis].astype(np.float32)
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)  # np.random.normal(mean, square, shape)
y_data = np.square(x_data) - 0.5 + noise


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


# 函数：获得batch

def next_batch(train_data, train_target, batch_size):
    index = [i for i in range(0, train_target.shape[0])]
    np.random.shuffle(index)
    batch_data = []
    batch_target = []
    for i in range(0, batch_size):
        batch_data = np.r_[batch_data, train_data[index[i]]]
        batch_target = np.r_[batch_target, train_target[index[i]]]
    if np.ndim(batch_data) == 1:
        batch_data = batch_data[:, np.newaxis]
    if np.ndim(batch_target) == 1:
        batch_target = batch_target[:, np.newaxis]
    return batch_data.astype(np.float32), batch_target.astype(np.float32)


# 定义网络结构
with tf.name_scope('inputs'):  # 名字象征

    x = tf.placeholder(tf.float32, [None, 1], name='x_input')
    y_ = tf.x = tf.placeholder(tf.float32, [None, 1], name='y_input')

layer_1 = add_layer(x, x.shape[1].value, 10, 1, tf.nn.relu)

layer_output = add_layer(layer_1, layer_1.shape[1].value, 1, 2)

y = layer_output

'''
# 交叉熵只能用于one-hot类型
# 这里的标签不是one-hot类型，我们要做的也不是多分类问题，所以不能采用交叉熵

loss = tf.reduce_sum(
    tf.nn.softmax_cross_entropy_with_logits_v2(_sentinel=None,
                                               labels=y_data,
                                               logits=y,
                                               name='cross_entropy'))
'''

# 定义 LOSS
with tf.name_scope('loss'):

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_),
                                        axis=1))
    # 此处axis=1意思是按列相加，即每列第一个数等于此列所有行之和

    tf.summary.scalar(':', loss)  # 显示loss在 Tensorboard 的 EVENT 中
    tf.summary.histogram(':', loss)

optimizer = tf.train.AdamOptimizer(0.1)


# 定义 train_step
with tf.name_scope('train_step'):
    train_step = optimizer.minimize(loss)

run_step = tf.tuple([y, loss, train_step])

''' 
error = tf.reduce_mean(tf.square(y_ - y))
'''


# 建立会话

with tf.Session() as inter_sess:

    merged = tf.summary.merge_all()  # 合并历史记录

    writer = tf.summary.FileWriter("logs/", inter_sess.graph)

    inter_sess.run(tf.global_variables_initializer())

    fig = plt.figure()  # 生成一个图片框
    ax = fig.add_subplot(1, 1, 1)  # 连续型画图
    ax.scatter(x_data, y_data)
    plt.ion()  # 由于存在plt.pause(0.1)命令，会暂停绘图；
    # plt.ion(): 连续绘图命令,能够在暂停后继续执行plt绘图，


    # y_eval = y.eval()
    print('loss for 50 steps:')

    for step in range(1000):

        batch_x, batch_y = next_batch(x_data, y_data, 30)

        run_step_outputs = inter_sess.run(run_step, feed_dict={x: x_data, y_: y_data})

        if step % 50 == 0:
            print(run_step_outputs[1])
            # print(inter_sess.run(loss))

            result = inter_sess.run(merged, feed_dict={x: x_data, y_: y_data})
            writer.add_summary(result, step)

            try:
                ax.lines.remove(lines[0])  # 抹除图片中lines的第一条线
            except Exception:  # Exception：捕获所有异常
                pass
    
            y = run_step_outputs[0]

            lines = ax.plot(x_data, y, 'r-', lw=5)  # lines 为点的连线
            plt.pause(0.1)  # 维持0.1秒后暂停



    plt.show()

    """
    plt.figure(1)
    plt.scatter(x_data, y_data, c='r', s=20, alpha=0.5)
    plt.plot(x_data, y.eval(), color='b', linewidth=1, alpha=0.8)
    plt.show()    
    
    """






