import tensorflow as tf
import numpy as np

# 创造一些data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### 创造tensorflow结构 ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)  # 一般是一个小于1的数字

train = optimizer.minimize(loss)


### 创造tensorflow结构 ###

# 创建一个会话
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)  # 初始化是必要的

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, ':', sess.run([Weights, biases]))

