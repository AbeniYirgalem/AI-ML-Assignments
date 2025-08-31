import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

print("TensorFlow version:", tf.__version__)

# ---------------------------------------------------
# Example: simple 3x3 image for convolution
# ---------------------------------------------------
image = np.array([[[[1], [2], [3]],
                   [[4], [5], [6]],
                   [[7], [8], [9]]]], dtype=np.float32)

print("image.shape:", image.shape)
plt.imshow(image.reshape(3, 3), cmap='gray')
plt.title("Original 3x3 Image")
plt.show()

# VALID padding
weight = tf.constant([[[[1.]], [[1.]]],
                      [[[1.]], [[1.]]]])
print("weight.shape:", weight.shape)

conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="VALID")
conv2d_img = conv2d.numpy()
print("conv2d_img.shape (VALID):", conv2d_img.shape)

for i, one_img in enumerate(np.swapaxes(conv2d_img, 0, 3)):
    plt.subplot(1, 2, i+1)
    plt.imshow(one_img.reshape(2, 2), cmap='gray')
plt.show()

# SAME padding
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="SAME")
conv2d_img = conv2d.numpy()
print("conv2d_img.shape (SAME):", conv2d_img.shape)

for i, one_img in enumerate(np.swapaxes(conv2d_img, 0, 3)):
    plt.subplot(1, 2, i+1)
    plt.imshow(one_img.reshape(3, 3), cmap='gray')
plt.show()

# ---------------------------------------------------
# Convolution with multiple filters
# ---------------------------------------------------
weight = tf.constant([[[[1., 10., -1.]], [[1., 10., -1.]]],
                      [[[1., 10., -1.]], [[1., 10., -1.]]]])
print("weight.shape:", weight.shape)

conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="SAME")
conv2d_img = conv2d.numpy()
print("conv2d_img.shape (multi-filter):", conv2d_img.shape)

for i, one_img in enumerate(np.swapaxes(conv2d_img, 0, 3)):
    plt.subplot(1, 3, i+1)
    plt.imshow(one_img.reshape(3, 3), cmap='gray')
plt.show()

# ---------------------------------------------------
# Max Pooling example
# ---------------------------------------------------
image2 = np.array([[[[4], [3]],
                    [[2], [1]]]], dtype=np.float32)

pool = tf.nn.max_pool(image2, ksize=[1, 2, 2, 1],
                      strides=[1, 1, 1, 1], padding='SAME')
print("MaxPool result:\n", pool.numpy())

# ---------------------------------------------------
# MNIST Example
# ---------------------------------------------------
(trainX, _), (_, _) = mnist.load_data()

img = trainX[132].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.title("MNIST Sample")
plt.show()

# Add channel dimension
img = img.reshape(-1, 28, 28, 1).astype('float32')

# Convolution with 5 filters
W1 = tf.Variable(tf.random.normal([3, 3, 1, 5], stddev=0.01))
conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')
conv2d_img = conv2d.numpy()
print("Conv2D MNIST output shape:", conv2d_img.shape)

for i, one_img in enumerate(np.swapaxes(conv2d_img, 0, 3)):
    plt.subplot(1, 5, i+1)
    plt.imshow(one_img.reshape(14, 14), cmap='gray')
plt.show()

# MaxPooling on convolution result
pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')
pool_img = pool.numpy()
print("MaxPool MNIST output shape:", pool_img.shape)

for i, one_img in enumerate(np.swapaxes(pool_img, 0, 3)):
    plt.subplot(1, 5, i+1)
    plt.imshow(one_img.reshape(7, 7), cmap='gray')
plt.show()
