import tensorflow as tf

mat_1 = tf.constant([[3, 3]])
mat_2 = tf.constant([[2],
                     [2]])

product = tf.matmul(mat_1, mat_2)

# 形式1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# 形式2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

