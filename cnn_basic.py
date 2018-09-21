import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
image = np.array([[[[1], [2], [3]],
                   [[4], [5], [6]],
                   [[7], [8], [9]]]], dtype=np.float32)

### Convolution ###

# 1 filter (2, 2, 1, 1) with padding:VALID
print("### 1filter (2, 2, 1, 1) with padding:VALID ###")

print("image.shape : ", image.shape)
plt.imshow(image.reshape(3, 3), cmap="Greys")
print("image.reshape(3, 3) : ", image.reshape(3, 3))

weight = tf.constant([[[[1.]], [[1.]]],
                      [[[1.]], [[1.]]]])
print("weight.shape", weight.shape)

conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="VALID")
conv2d_img = conv2d.eval()
print("conv2d_img.shape : ", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
print("conv2d_img.shape(swapaxes) : ", conv2d_img.shape)

for i, one_img in enumerate(conv2d_img):
    print("one_img : \n", one_img)
    print("one_img.reshape(2, 2) : \n", one_img.reshape(2, 2))
    plt.subplot(1, 2, i + 1), plt.imshow(one_img.reshape(2, 2), cmap="gray")
plt.waitforbuttonpress()

# 1 filter (2, 2, 1, 1) with padding:SAME
print("### 1filter (2, 2, 1, 1) with padding:SAME ###")

print("image.shape : ", image.shape)

weight = tf.constant([[[[1.]], [[1.]]],
                      [[[1.]], [[1.]]]])
print("weight.shape", weight.shape)

conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="SAME")
conv2d_img = conv2d.eval()
print("conv2d_img.shape : ", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
print("conv2d_img.shape(swapaxes) : ", conv2d_img.shape)

for i, one_img in enumerate(conv2d_img):
    print("one_img : \n", one_img)
    print("one_img.reshape(3, 3) : \n", one_img.reshape(3, 3))
    plt.subplot(1, 2, i + 1), plt.imshow(one_img.reshape(3, 3), cmap="gray")
plt.waitforbuttonpress()

# 3 filters (2, 2, 1, 3) with padding:SAME
print("### 3 filters (2, 2, 1, 3) with padding:SAME ###")

print("image.shape : ", image.shape)

weight = tf.constant([[[[1., 10., -1.]], [[1., 10., -1.]]],
                      [[[1., 10., -1.]], [[1., 10., -1.]]]])
print("weight.shape", weight.shape)

conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="SAME")
conv2d_img = conv2d.eval()
print("conv2d_img.shape : ", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
print("conv2d_img.shape(swapaxes) : ", conv2d_img.shape)

for i, one_img in enumerate(conv2d_img):
    print("one_img : \n", one_img)
    print("one_img.reshape(3, 3) : \n", one_img.reshape(3, 3))
    plt.subplot(1, 3, i + 1), plt.imshow(one_img.reshape(3, 3), cmap="gray")
plt.waitforbuttonpress()





### POOLING ###

# MAX POOLING with padding:VALID
image = np.array([[[[4],[3]],
                    [[2],[1]]]], dtype=np.float32)
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                    strides=[1, 1, 1, 1], padding='VALID')
print("pool.shape : ", pool.shape)
print("pool.eval() : \n", pool.eval())

# MAX POOLING with padding:SAME
image = np.array([[[[4],[3]],
                    [[2],[1]]]], dtype=np.float32)
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                    strides=[1, 1, 1, 1], padding='SAME')
print("pool.shape : ", pool.shape)
print("pool.eval() : \n", pool.eval())





### mnist img conv & pool ###

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

img = mnist.train.images[0].reshape(28,28)
plt.imshow(img, cmap='gray')

# 28 x 28 x 1 의 이미지 여러장.
img = img.reshape(-1,28,28,1)
# 3 x 3 x 1 의 필터 5 개.
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))
# strides 가 2 x 2, padding:SAME 이므로 28 x 28 x 1 -> 14 x 14 x 5 가 될 것.
conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')
print(conv2d)
sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')
plt.waitforbuttonpress()

# strides 가 2 x 2, padding:SAME 이므로 7 x 7 x 5 가 될 것
pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(pool)
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7, 7), cmap='gray')
plt.waitforbuttonpress()