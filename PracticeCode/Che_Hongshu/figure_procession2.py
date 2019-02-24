import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile(
    'F:/aProject_EN/TensorFlowLearning/PracticeCode/Che_Hongshu/TempFile/timg.jpg',
    'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    plt.imshow(img_data.eval())
    plt.show()

    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    img_resize = tf.image.resize_images(img_data, [200, 200], method=0)
    plt.imshow(img_resize.eval())
    #####
    plt.show()

    img_data = tf.image.convert_image_dtype(img_resize, dtype=tf.uint8)

    encode_image = tf.image.encode_jpeg(img_data)

    with tf.gfile.GFile('./New-Woman.jpg', 'wb') as f:
        f.write(encode_image.eval())
