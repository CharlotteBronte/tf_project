#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@desc:   简单的skip-gram模型实现的word2vec
@time:   2017/06/19 20：48
@author: liuluxin(0_0mirror@sina.com)
@param:
@param:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import sys, os
import random
import zipfile
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import ConfigParser

line_split = ""
qa_split = ""
word_split = ""
config_file = "/word_embedding.conf"

'''
@desc: 得到路径配置
@format: [word2vec] train_data=xxxx
'''
def get_config(section, key):
    config = ConfigParser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0] + config_file
    config.read(path)
    return config.get(section, key)

'''
@desc: 得到路径配置
@format: [word2vec] train_data=xxxx
'''
def get_config_int(section, key):
    config = ConfigParser.ConfigParser()
    path = os.path.split(os.path.realpath(__file__))[0] + config_file
    config.read(path)
    return config.get_int(section, key)


'''
@desc: 从行从split出word并将低频词和停用词(删除概率P(Wi)=1-sqrt(t/frequent(Wi))都平滑掉
@param: freq:小于freq次数的词都会被删除
        del_threshold: 大于这个阈值的词会被作为停用词被删除
'''
def build_dict(freq=5, del_threshold=1e-5):
    raw_words = open(get_config("word2vec","train_data")).replace(qa_split, word_split).replace(line_split, word_split).split(word_split)
    word_counts = Counter(raw_words)
    # 计算总词频
    total_count = len(raw_words)
    word_freq = {w: c / total_count for w, c in word_counts.items()}
    prob_drop = {w: 1 - np.sqrt(del_threshold / f) for w, f in word_freq.items()}
    # 将低频和停用词都剔除成为训练数据，被剔除的使用UNK做平滑
    train_words = [w for w in raw_words if (prob_drop[w]<del_threshold and word_counts[w]>freq)]
    trimed_dict = {w:0 for w in raw_words if (prob_drop[w]>=del_threshold or word_counts[w]<=freq)}
    vocab = set(train_words)
    vocab.add("UNK")
    vocab_2_idx = {w: c for c, w in enumerate(vocab)}
    idx_2_vocab = {c: w for c, w in enumerate(vocab)}
    print("Total words:{}".format(len(train_words)))
    print("Unique words:{}".format(len(train_words)))
    print("Trimed words:{}".format(len(trimed_dict)))
    return vocab_2_idx, idx_2_vocab

dictionary, reverse_dictionary = build_dict(vocabulary, vocabulary_size)


'''
@desc: 每次调用从每行中随机产生batch数据
@param: batch_size: 每次扫描的块的大小，
        num_skips:每个词的重用次数，取决于window的大小
        skip_window: 采样词的左右窗口大小（即决定了进行几gram的采样)skip_windows*2=num_skips
'''
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


'''
@desc: 保存模型到配置路径
'''
def save_model():
    save_path = get_config("word2vec","model_path")
    if os.path.isfile(save_path):
        raise RuntimeError('the save path should be a dir')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        self.saver.save(self.sess,tf_path)



batch_size = get_config_int("word2vec","batch_size")
embedding_size = get_config_int("word2vec","embedding_size")  # Dimension of the embedding vector.
skip_window = get_config_int("word2vec","skip_window")  # How many words to consider left and right.
num_skips = get_config_int("word2vec","num_skips")  # How many times to reuse an input to generate a label.


valid_size = get_config_int("word2vec","valid_size")  # Random set of words to evaluate similarity on.
valid_window = get_config_int("word2vec","valid_window")  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = get_config_int("word2vec","num_sampled")   # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/gpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),name="embeddings")
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)), name="nec_weight")
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]),name="nce_biases")

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
    merged_summary_op = tf.merge_all_summaries()

    # Add variable initializer.


'''
正式训练流程
'''
num_steps = get_config_int("word2vec", "num_steps")
#初始化所有数据和存储
saver = tf.train.Saver()
log_dir =  get_config("word2vec","word2vec_log")
print('Begin Training')
with tf.Session(graph=graph) as session:
    model_path = get_config("word2vec","model_path")
    # 存在就从模型中恢复变量
    if os.path.exists(model_path):
        saver.restore(sess, model_path)
    # 不存在就初始化变量
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
    summary_writer = tf.train.SummaryWriter(log_dir, session.graph)


    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        tf.scalar_summary('loss', loss_val)
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        # 每隔100次迭代，保存一次日志
        summary_str = session.run(merged_summary_op)
        summary_writer.add_summary(summary_str, step)

        # 每2000次迭代输出一次损失 保存一次模型
        if step % get_config_int == 0:
            if step > 0:
                average_loss /= 2000
            average_loss = 0
            save_path = saver.save(sess, model_path)
            print("模型保存:{0}\t当前损失:{1}".format(model_path,average_loss))

        # 每隔迭代输出一次指定词语的最近邻居

        if step % 100 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()


'''
@desc:绘制图像存储到指定png文件
'''
def plot_with_labels(low_dim_embs, labels, filename='lyric_word_vec.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure()  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label.encode("utf-8"),
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom', fontproperties=myfont)

    plt.savefig(filename)
    #
    # file = open('lyric_word_vec.png', 'rb')
    # data = file.read()
    # file.close()
    # # 图片处理
    # image = tf.image.decode_png(data, channels=4)
    # image = tf.expand_dims(image, 0)
    #
    # # 添加到日志中
    # sess = tf.Session()
    # writer = tf.summary.FileWriter('logs')
    # summary_op = tf.summary.image("image1", image)
    #
    # # 运行并写入日志
    # summary = sess.run(summary_op)
    # writer.add_summary(summary)


try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib as mpl

    mpl.use('Agg')
    from matplotlib.font_manager import *
    import matplotlib.pyplot as plt

    myfont = FontProperties(
        fname="//data01/ai_rd/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/msyh.ttf")
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    embs_pic_path = get_config("word2vec", "embs_pic_path")
    plot_with_labels(low_dim_embs, labels, embs_pic_path)
except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
