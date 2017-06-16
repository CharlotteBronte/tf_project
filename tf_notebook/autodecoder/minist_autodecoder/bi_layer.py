#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@desc:  两个隐层的自编码器
@time:   2017/06/16 10：07
@author: lucy(0_0mirror@sina.com)
@param:
@output:#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 128  # batch容量
display_step = 1  # 展示间隔
learning_rate = 0.01  # 学习率
training_epochs = 20  # 训练轮数，1轮等于n_samples/batch_size
example_to_show = 10  # 展示图像数目

n_hidden1_units = 256  # 第一隐藏层
n_hidden2_units = 128  # 第二隐藏层
n_input_units = 784
n_output_units = n_input_units


def variable_summaries(var):  # <---
    """
    可视化变量全部相关参数
    :param var:
    :return:
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.histogram('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)  # 注意，这是标量
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


def WeightsVariable(n_in, n_out, name_str):
    return tf.Variable(tf.random_normal([n_in, n_out]), dtype=tf.float32, name=name_str)


def biasesVariable(n_out, name_str):
    return tf.Variable(tf.random_normal([n_out]), dtype=tf.float32, name=name_str)


def encoder(x_origin, activate_func=tf.nn.sigmoid):
    with tf.name_scope('Layer1'):
        Weights = WeightsVariable(n_input_units, n_hidden1_units, 'Weights')
        biases = biasesVariable(n_hidden1_units, 'biases')
        x_code1 = activate_func(tf.add(tf.matmul(x_origin, Weights), biases))
        variable_summaries(Weights)  # <---
        variable_summaries(biases)  # <---
    with tf.name_scope('Layer2'):
        Weights = WeightsVariable(n_hidden1_units, n_hidden2_units, 'Weights')
        biases = biasesVariable(n_hidden2_units, 'biases')
        x_code2 = activate_func(tf.add(tf.matmul(x_code1, Weights), biases))
        variable_summaries(Weights)  # <---
        variable_summaries(biases)  # <---
    return x_code2


def decode(x_code, activate_func=tf.nn.sigmoid):
    with tf.name_scope('Layer1'):
        Weights = WeightsVariable(n_hidden2_units, n_hidden1_units, 'Weights')
        biases = biasesVariable(n_hidden1_units, 'biases')
        x_decode1 = activate_func(tf.add(tf.matmul(x_code, Weights), biases))
        variable_summaries(Weights)  # <---
        variable_summaries(biases)  # <---
    with tf.name_scope('Layer2'):
        Weights = WeightsVariable(n_hidden1_units, n_output_units, 'Weights')
        biases = biasesVariable(n_output_units, 'biases')
        x_decode2 = activate_func(tf.add(tf.matmul(x_decode1, Weights), biases))
        variable_summaries(Weights)  # <---
        variable_summaries(biases)  # <---
    return x_decode2


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

    # 标量汇总
    with tf.name_scope('LossSummary'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)

    # 图像展示
    with tf.name_scope('ImageSummary'):
        image_original = tf.reshape(X_input, [-1, 28, 28, 1])
        image_reconstruction = tf.reshape(X_decode, [-1, 28, 28, 1])
        tf.summary.image('image_original', image_original, 9)
        tf.summary.image('image_recinstruction', image_reconstruction, 9)

    # 汇总
    merged_summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

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
                summary_str = sess.run(merged_summary, feed_dict={X_input: batch_xs})  # <---
                writer.add_summary(summary_str, epoch)  # <---
                writer.flush()  # <---
        writer.close()
        print('训练完毕！')