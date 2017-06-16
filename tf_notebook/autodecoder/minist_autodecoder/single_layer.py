#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@desc:  单层自编码器
@time:   2017/06/15 12:40
@author: lucy(0_0mirror@sina.com)
@param:
@output:#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate = 0.01  # 学习率
training_epochs = 20  # 训练轮数，1轮等于n_samples/batch_size
batch_size = 128  # batch容量
display_step = 1  # 展示间隔
example_to_show = 10  # 展示图像数目

n_hidden_units = 256
n_input_units = 784
n_output_units = n_input_units


def WeightsVariable(n_in, n_out, name_str):
    return tf.Variable(tf.random_normal([n_in, n_out]), dtype=tf.float32, name=name_str)


def biasesVariable(n_out, name_str):
    return tf.Variable(tf.random_normal([n_out]), dtype=tf.float32, name=name_str)


def encoder(x_origin, activate_func=tf.nn.sigmoid):
    with tf.name_scope('Layer'):
        Weights = WeightsVariable(n_input_units, n_hidden_units, 'Weights')
        biases = biasesVariable(n_hidden_units, 'biases')
        x_code = activate_func(tf.add(tf.matmul(x_origin, Weights), biases))
    return x_code


def decode(x_code, activate_func=tf.nn.sigmoid):
    with tf.name_scope('Layer'):
        Weights = WeightsVariable(n_hidden_units, n_output_units, 'Weights')
        biases = biasesVariable(n_output_units, 'biases')
        x_decode = activate_func(tf.add(tf.matmul(x_code, Weights), biases))
    return x_decode


with tf.Graph().as_default():
    with tf.name_scope('Input'):
        X_input = tf.placeholder(tf.float32, [None, n_input_units])
    with tf.name_scope('Encode'):
        X_code = encoder(X_input)
    with tf.name_scope('decode'):
        X_decode = decode(X_code)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.pow(X_input - X_decode, 2))
    with tf.name_scope('train'):
        Optimizer = tf.train.RMSPropOptimizer(learning_rate)
        train = Optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    # 因为使用了tf.Graph.as_default()上下文环境
    # 所以下面的记录必须放在上下文里面，否则记录下来的图是空的（get不到上面的default）
    writer = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())
    writer.flush()

    mnist = input_data.read_data_sets('../Mnist_data/', one_hot=True)

    with tf.Session() as sess:
        sess.run(init)
        total_batch = int(mnist.train.num_examples / batch_size)
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, Loss = sess.run([train, loss], feed_dict={X_input: batch_xs})
                Loss = sess.run(loss, feed_dict={X_input: batch_xs})
            if epoch % display_step == 0:
                print('Epoch: %04d' % (epoch + 1), 'loss= ', '{:.9f}'.format(Loss))
        writer.close()
        print('训练完毕！')

        '''比较输入和输出的图像'''
        # 输出图像获取
        reconstructions = sess.run(X_decode, feed_dict={X_input: mnist.test.images[:example_to_show]})
        # 画布建立
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(example_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(reconstructions[i], (28, 28)))
        f.show()  # 渲染图像
        plt.draw()  # 刷新图像
        # plt.waitforbuttonpress()