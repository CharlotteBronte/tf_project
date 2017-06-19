#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@desc:  滑动平均模型例子
@time:   2017/06/18 15：22
@author: lucy(0_0mirror@sina.com)
@param:
@output:
"""
import tensorflow as tf

v1 = tf.Variable(0, dtype=tf.float32)
step = tf.Variable(0, trainable=False)
#一些训练优化算法，比如GradientDescent 和Momentum 在优化过程中便可以使用到移动平均方法。
# 使用移动平均常常可以较明显地改善结果。
ema = tf.train.ExponentialMovingAverage(0.99, step)
maintain_averages_op = ema.apply([v1])
with tf.Session() as sess:
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print sess.run([v1, ema.average(v1)])

    # 更新变量v1的取值
    sess.run(tf.assign(v1, 5))
    sess.run(maintain_averages_op)
    print sess.run([v1, ema.average(v1)])

    # 更新step和v1的取值
    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_averages_op)
    print sess.run([v1, ema.average(v1)])

    # 更新一次v1的滑动平均值
    sess.run(maintain_averages_op)
    print sess.run([v1, ema.average(v1)])