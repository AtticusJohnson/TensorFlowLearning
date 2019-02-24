import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)


# 下载IMDB数据集
# 对句子进行多热编码，就是将句子转化为由0和1组成的向量
# 该模型将很快过拟合，它将被用来演示如何过拟合以及如何阻止它

NUM_WORDS = 10000


(train_data, train_labels), (test_data, test_labels) \
    = keras.datasets.imdb.load_data(num_words=NUM_WORDS)


def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequence), dimension)
    results = np.zeros((len(sequences), dimension))

    for i, word_indices in enumerate(sequences):
        # set specific indices of results[i] to 1s
        results[i, word_indices] = 1.0

    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

plt.plot(train_data[0])


# 创建基准模型

baseline_model = keras.Sequential([
    # 'input_shape' is only required here so that '.summary' works.
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()

baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)


# 创建一个更小的模型
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)])

smaller_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])

smaller_model.summary()

smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs=20,
                                    batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)


# 创建一个更大的模型
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy', 'binary_crossentropy'])

bigger_model.summary()

bigger_history = bigger_model.fit(train_data,
                                  train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)


# 绘制训练损失和验证损失图表
# 实线表示训练损失，虚线表示验证损失（验证损失越低，表示模型越好）
# 在此示例中，较小的网络开始过拟合的时间比基准模型晚（前者在6个周期之后，后者在4个周期之后）
# 并且，开始过拟合之后，其效果下降速度也慢得多

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['Val_'+key],
                       '--', label=name.title()+'Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+'Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('-', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])

    plot_history([('baseline', baseline_history),
                  ('smaller', smaller_history),
                  ('bigger', bigger_history)])

