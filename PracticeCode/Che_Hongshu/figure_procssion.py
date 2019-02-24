# 图像编码处理
# 彩色图片为RGB三个通道，所以可以看成一个三维矩阵，矩阵中的每一个
# 数字表示了图像上不同的位置，不同颜色的亮度。然而对于图片的存储，
# 并非直接存储这个矩阵，而是对图片进行了编码之后存储的

import __future__
import matplotlib.pyplot as plt
import tensorflow as tf


# 读取图像的原始图像  这里可能会出现decode‘utf-8’的error读用rb就搞定
# 读入的为二进制流，  ./yangmi.jpg 为当前程序文件夹的图片途径
# tf.gfile.FastGFile为tf自带的读取数据的操作函数
image_raw_data = tf.gfile.FastGFile(
    'F:/aProject_EN/TensorFlowLearning/PracticeCode/Che_Hongshu/TempFile/timg.jpg',
    'rb').read()

with tf.Session() as sess:
    # 对图片进行解码,二进制文件解码为uint8
    img_data = tf.image.decode_jpeg(image_raw_data)
    # 输出图片数据（三维矩阵）每个数字都为0-255之间的数字
    print(img_data.eval())
    plt.imshow(img_data.eval())
    plt.show()
    # 图片按照jpeg格式编码
    encode_image = tf.image.encode_jpeg(img_data)
    # 将图片转换为float32类型，相当于归一化，矩阵中的数字为0-1
    # 这个操作是对图片的大小调整等操作提供便利
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    # 创建文件并写入
    with tf.gfile.GFile('./woman', 'wb') as f:
        f.write(encode_image.eval())






