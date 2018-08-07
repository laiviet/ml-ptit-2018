import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, reshape=False)
batch_size = 2
example_X, example_ys = mnist.train.next_batch(batch_size)
example_X = tf.image.resize_images(example_X,[224,224])
session = tf.InteractiveSession()

def convolution(X, W, b, padding, stride):
    n, h, w, c = map(lambda d: d.value, X.get_shape())
    filter_h, filter_w, filter_c, filter_n = [d.value for d in W.get_shape()]

    out_h = (h + 2 * padding - filter_h) // stride + 1
    out_w = (w + 2 * padding - filter_w) // stride + 1

    X_flat = flatten(X, filter_h, filter_w, filter_c, out_h, out_w, stride, padding)
    W_flat = tf.reshape(W, [filter_h * filter_w * filter_c, filter_n])

    # print("X_flat: " + str(X_flat.shape) + " " + "W_flat: " + str(W_flat.shape))
    z = tf.matmul(X_flat, W_flat) + b  # b: 1 X filter_n
    return tf.transpose(tf.reshape(z, [out_h, out_w, n, filter_n]), [2, 0, 1, 3])

def flatten(X, window_h, window_w, window_c, out_h, out_w, stride=1, padding=0):
    X_padded = tf.pad(X, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
    windows = []
    for y in range(out_h):
        for x in range(out_w):
            window = tf.slice(X_padded, [0, y * stride, x * stride, 0], [-1, window_h, window_w, -1])
            windows.append(window)
    stacked = tf.stack(windows)  # shape : [out_h, out_w, n, filter_h, filter_w, c]

    return tf.reshape(stacked, [-1, window_c * window_w * window_h])

def relu(X):
    return tf.maximum(X, tf.zeros_like(X))

def max_pool(X, pool_h, pool_w, padding, stride):
    n, h, w, c = [d.value for d in X.get_shape()]

    out_h = (h + 2 * padding - pool_h) // stride + 1
    out_w = (w + 2 * padding - pool_w) // stride + 1

    X_flat = flatten(X, pool_h, pool_w, c, out_h, out_w, stride, padding)

    pool = tf.reduce_max(tf.reshape(X_flat, [out_h, out_w, n, pool_h * pool_w, c]), axis=3)
    return tf.transpose(pool, [2, 0, 1, 3])

#----------Convolution Layer 1----------#

X = tf.placeholder('float', [batch_size, 224, 224, 1])
t = tf.placeholder('float', [batch_size, 10])
filter_h, filter_w, filter_c, filter_n = 11, 11, 1, 96
W1 = tf.Variable(tf.random_normal([filter_h, filter_w, filter_c, filter_n], stddev=0.01))
b1 = tf.Variable(tf.zeros([filter_n]))
conv_layer_1 = convolution(X, W1, b1, padding=2, stride=4)
conv_activation_layer_1 = relu(conv_layer_1)
print("conv_1" + str(conv_activation_layer_1.shape))

# Max pooling
pooling_layer_1 = max_pool(conv_activation_layer_1, pool_h=3, pool_w=3, padding=0, stride=2)
print("pooling_1" + str(pooling_layer_1.shape))

#----------Convolution Layer 2----------#

filter_h, filter_w, filter_c, filter_n = 5, 5, 96, 256
W2 = tf.Variable(tf.random_normal([filter_h, filter_w, filter_c, filter_n], stddev=0.01))
b2 = tf.Variable(tf.zeros([filter_n]))
conv_layer_2 = convolution(pooling_layer_1, W2, b2, padding=2, stride=1)
conv_activation_layer_2 = relu(conv_layer_2)
print("conv_2" + str(conv_activation_layer_2.shape))

# Max pooling
pooling_layer_2 = max_pool(conv_activation_layer_2, pool_h=3, pool_w=3, padding=0, stride=2)
print("pooling_2" + str(pooling_layer_2.shape))

#----------Convolution Layer 3----------#

filter_h, filter_w, filter_c, filter_n = 3, 3, 256, 384
W3 = tf.Variable(tf.random_normal([filter_h, filter_w, filter_c, filter_n], stddev=0.01))
b3 = tf.Variable(tf.zeros([filter_n]))
conv_layer_3 = convolution(pooling_layer_2, W3, b3, padding=1, stride=1)
conv_activation_layer_3 = relu(conv_layer_3)
print("conv_3" + str(conv_activation_layer_3.shape))

#----------Convolution Layer 4----------#

filter_h, filter_w, filter_c, filter_n = 3, 3, 384, 384
W4 = tf.Variable(tf.random_normal([filter_h, filter_w, filter_c, filter_n], stddev=0.01))
b4 = tf.Variable(tf.zeros([filter_n]))
conv_layer_4 = convolution(conv_activation_layer_3, W4, b4, padding=1, stride=1)
conv_activation_layer_4 = relu(conv_layer_4)
print("conv_4" + str(conv_activation_layer_4.shape))

#----------Convolution Layer 5----------#

filter_h, filter_w, filter_c, filter_n = 3, 3, 384, 256
W4 = tf.Variable(tf.random_normal([filter_h, filter_w, filter_c, filter_n], stddev=0.01))
b4 = tf.Variable(tf.zeros([filter_n]))
conv_layer_4 = convolution(conv_activation_layer_3, W4, b4, padding=1, stride=1)
conv_activation_layer_4 = relu(conv_layer_4)
print("conv_5" + str(conv_activation_layer_4.shape))

# Max pooling
pooling_layer_3 = max_pool(conv_activation_layer_4, pool_h=3, pool_w=3,padding=0, stride=2)
print("pooling_3" + str(pooling_layer_3.shape))

batch_size, pool_output_h, pool_output_w, filter_n = [d.value for d in pooling_layer_3.get_shape()]
hidden_size = 4096
W5 = tf.Variable(tf.random_normal([pool_output_h * pool_output_w * filter_n, hidden_size], stddev=0.01))
b5 = tf.Variable(tf.zeros([hidden_size]))


def affine(X, W, b):
    n = X.get_shape()[0].value  # number of samples
    X_flat = tf.reshape(X, [n, -1])
    return tf.matmul(X_flat, W) + b


affine_layer1 = affine(pooling_layer_3, W5, b5)
init = tf.global_variables_initializer()
init.run()
example_X = session.run(example_X)
affine_layer1.eval({X: example_X, t: example_ys})[0]
affine_activation_layer1 = relu(affine_layer1)
affine_activation_layer1.eval({X: example_X, t: example_ys})[0]
output_size = 10
W6 = tf.Variable(tf.random_normal([hidden_size, output_size], stddev=0.01))
b6 = tf.Variable(tf.zeros([output_size]))
affine_layer2 = affine(affine_activation_layer1, W6, b6)
init = tf.global_variables_initializer()
init.run()
affine_layer2.eval({X: example_X, t: example_ys})[0]


def softmax(X):
    X_centered = X - tf.reduce_max(X)  # to avoid overflow
    X_exp = tf.exp(X_centered)
    exp_sum = tf.reduce_sum(X_exp, axis=1)
    return tf.transpose(tf.transpose(X_exp) / exp_sum)


softmax_layer = softmax(affine_layer2)
softmax_layer.eval({X: example_X, t: example_ys})[0]


def cross_entropy_error(y, t):
    return -tf.reduce_mean(tf.log(tf.reduce_sum(y * t, axis=1)))


loss = cross_entropy_error(softmax_layer, t)
loss.eval({X: example_X, t: example_ys})
learning_rate = 0.1
trainer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
training_epochs = 2
num_batch = int(mnist.train.num_examples / batch_size)
from tqdm import tqdm_notebook
test_x = mnist.test.images[:batch_size]
test_t = mnist.test.labels[:batch_size]
test_x = tf.image.resize_images(test_x,[224,224])
test_x = session.run(test_x)

def accuracy(network, t):
    t_predict = tf.argmax(network, axis=1)
    t_actual = tf.argmax(t, axis=1)

    return tf.reduce_mean(tf.cast(tf.equal(t_predict, t_actual), tf.float32))
for epoch in range(training_epochs):
    avg_cost = 0
    for _ in tqdm_notebook(range(num_batch)):
        train_X, train_ys = mnist.train.next_batch(batch_size)
        train_X = session.run(tf.image.resize_images(train_X, [224, 224]))
        trainer.run(feed_dict={X: train_X, t: train_ys})
        avg_cost += loss.eval(feed_dict={X: train_X, t: train_ys}) / num_batch

    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost), "accuracy=", "{:.9f}".format(accuracy(softmax_layer, t).eval(feed_dict={X: test_x, t: test_t})), flush=True)

session.close()