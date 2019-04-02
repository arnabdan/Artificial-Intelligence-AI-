from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib.pyplot as plt
import tensorflow as tf

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None
mnist = input_data.read_data_sets("/tmp/MNIST_data/", one_hot=True)
batch_x1, batch_y1 =mnist.train.next_batch(10)

x_i = batch_x1[0]
print("----------------")
print(x_i.shape)
print(x_i)
first_array=batch_x1[0]
image = first_array.reshape((28, 28))
print("=========")
print(image)

plt.imshow(image)
plt.show()

batch_x1, batch_y1 =mnist.train.next_batch(2)

##Display image
label = batch_y1[1]
print("=====Y1---------------====")
print(batch_y1[0])

print("=====X1---------------====")
print(batch_x1[0])

pixels = batch_x1[1]
#pixels = np.array(pixels, dtype='uint8')
pixels = pixels.reshape((28, 28))
plt.title('Label is {label}'.format(label=label))
plt.imshow(pixels, cmap='gray')
plt.show()

