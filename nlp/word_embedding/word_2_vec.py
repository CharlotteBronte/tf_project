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
from collections import Counter
import math
import sys, os
import random
import zipfile
import sys
import pickle

reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import ConfigParser

line_split = ""
qa_split = ""
word_split = ""
config_file = "/word_embedding.ini"

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
    return config.getint(section, key)

#从原始文件中得到词表并存储在csv文件中
def read_raw_words():
    raw_words = open(get_config("word2vec","train_data")).readline().replace(qa_split, word_split).replace(line_split, word_split).split(word_split)
    raw_file = open(get_config("word2vec","raw_words_file"), 'wb')
    pickle.dump(raw_words, raw_file)
    raw_file.close()
    print("Raw words:{}".format(len(raw_words)))

'''
@desc: 从行从split出word并将低频词和停用词(删除概率P(Wi)=1-sqrt(t/frequent(Wi))都平滑掉
@param: freq:小于freq次数的词都会被删除
        del_threshold: 大于这个阈值的词会被作为停用词被删除
'''
def build_dict(freq=3, del_threshold=0.95):
    read_raw_words()
    raw_words_file = open(get_config("word2vec", "raw_words_file"),"rb")
    raw_words = pickle.load(raw_words_file)
    raw_words_file.close()
    print("读取文件成功，总词表长度为{0}".format(len(raw_words)))

    word_counts = Counter(raw_words)
    # 计算总词频
    total_count = len(raw_words)
    word_freq = {w: c / total_count for w, c in word_counts.items()}
    prob_drop = {w: 1 - np.sqrt(1e-5/f) for w, f in word_freq.items()}
   
    # 将低频和停用词都剔除成为训练数据，被剔除的使用UNK做平滑
    train_words = [w for w in raw_words if(prob_drop[w]<del_threshold and word_counts[w]>freq)]
    trimed_dict = {w:0 for w in raw_words if (prob_drop[w]>=del_threshold or word_counts[w]<=freq)}
    vocab = set(train_words)
    vocab.add("UNK")
    vocab_2_idx = {w: c for c, w in enumerate(vocab)}
    idx_2_vocab = {c: w for c, w in enumerate(vocab)}
    print("Total words:{}".format(len(raw_words)))
    print("Unique words:{}".format(len(train_words)))
    print("Trimed words:{}".format(len(trimed_dict)))
    return  vocab_2_idx, idx_2_vocab, trimed_dict

vocab_2_idx, idx_2_vocab, trimed_dictionary = build_dict()


'''
@desc: 将所有的原始数据idx化并存储
@param: qa_file_name:对话数据文件路径
        idx_file_name: 索引化后的文件路径
'''
def build_sent_file(qa_file_name, idx_file_name):
    qa_sents = open(get_config("word2vec", "train_data")).readline().split(line_split)
    q_list = []
    a_list = []
    # 对每个qasent进行分词然后逐个得到batch信息
    for sent_pair in qa_sents:
        sent_list = sent_pair.split(qa_split)
        if len(sent_list) != 2:
            continue;
        # list向量化
        q_list.append(map(lambda x: vocab_2_idx["UNK"] if x in trimed_dictionary.keys() else vocab_2_idx[x],
                          single_sent.split(word_split), sent_list[0].split(word_split)))
        a_list.append(map(lambda x: vocab_2_idx["UNK"] if x in trimed_dictionary.keys() else vocab_2_idx[x],
                          single_sent.split(word_split), sent_list[1].split(word_split)))
    sent_vec_file = open(get_config("word2vec", "sent_vec_file"), 'wb')
    pickle.dump(q_list, sent_vec_file)
    pickle.dump(a_list, sent_vec_file)
    sent_vec_file.close()


build_sent_file(get_config("word2vec", "qa_file_name"), get_config("word2vec", "idx_file_name"))
idx_file = open(get_config("word2vec", "idx_file_name"), "rb")
q_sents = pickle.load(idx_file)
a_sents = pickle.load(idx_file)
idx_file.close();
batch_nums = len(batchs)


'''
@desc: 从qa文件的每行中，在windowsize的窗口内随机产生batch数据
@param: line_begin: 开始采样的句子位置
        line_end: 结束采样的句子位置
        num_skips:每个词的重用次数，取决于window的大小
        skip_window: 采样词的左右窗口大小（即决定了进行几gram的采样)skip_windows*2=num_skips
'''
def generate_batch(line_begin, line_end, num_skips, skip_window):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    UNK_idx = vocab_2_idx["UNK"]
    batch_list = []
    label_list = []
    #根据指定的行号从q和a的sentence中取出需要的batch
    for line_idx in range(line_begin,line_end):
        for line_type in range(0,2):
            if line_type==1:
                query_list = q_sents[line_idx]
            else:
                query_list = a_sents[line_idx]

            for idx in range(len(query_list)):
                if query_list[idx] != UNK_idx:
                    input_id = query_list[idx]
                    target_window = np.random.randint(1, skip_window + 1)
                    start = max(0, idx - target_window)
                    end = min(len(query_list) - 1, idx + target_window)
                for i in range(start, end):
                    if idx != i:
                        output_id = query_list[i]
                        batch_list.append(input_id)
                        label_list.append(output_id)
    return batch_list,label_list


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], idx_2_vocab[batch[i]],
          '->', labels[i, 0], idx_2_vocab[labels[i, 0]])


'''
@desc: 从配置中读取，并构建图所需的元素
'''
batch_size = get_config_int("word2vec", "batch_size")
embedding_size = get_config_int("word2vec", "embedding_size")  # Dimension of the embedding vector.
skip_window = get_config_int("word2vec", "skip_window")  # How many words to consider left and right.
num_skips = get_config_int("word2vec", "num_skips")  # How many times to reuse an input to generate a label.

valid_size = get_config_int("word2vec", "valid_size")  # Random set of words to evaluate similarity on.
valid_window = get_config_int("word2vec", "valid_window")  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = get_config_int("word2vec", "num_sampled")  # Number of negative examples to sample.

graph = tf.Graph()
with graph.as_default():
    with tf.device('/gpu:0'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="embeddings")
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)), name="nec_weight")
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="nce_biases")

    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # 计算候选embedding的cosine相似度
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
    merged_summary_op = tf.summary.merge_all()


'''
@desc: 正式训练流程
'''
num_steps = get_config_int("word2vec", "num_steps")
#初始化所有数据和存储
log_dir =  get_config("word2vec","log_dir")
print('Begin Training')
with tf.Session(graph=graph) as session:
    model_path = get_config("word2vec","model_path")
    saver = tf.train.Saver()
    # 存在就从模型中恢复变量
    if os.path.exists(model_path):
        saver.restore(session, model_path)
    # 不存在就初始化变量
    else:
        init = tf.global_variables_initializer()
        session.run(init)
    summary_writer = tf.summary.FileWriter(log_dir, session.graph)

    average_loss = 0
    for step in xrange(num_steps):
        begin_idx = step * batch_size % batch_nums
        end_idx = (step + 1) * batch_size % batch_nums
        batch_inputs = []
        batch_labels = []
        if end_idx < begin_idx:
            inputs,labels = generate_batch(0, end_idx, num_skips, skip_window)
            batch_inputs.extend(inputs)
            batch_labels.extend(labels)
            inputs, labels = generate_batch(begin_idx, batch_nums, num_skips, skip_window)
            batch_inputs.extend(inputs)
            batch_labels.extend(labels)
        else:
            inputs, labels = generate_batch(begin_idx, end_idx, num_skips, skip_window)
            batch_inputs.extend(inputs)
            batch_labels.extend(labels)

        input_size =len(batch_labels)
        train_inputs = tf.placeholder(tf.int32, shape=[input_size])
        train_labels = tf.placeholder(tf.int32, shape=[input_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        tf.summary.scalar('loss',average_loss)
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        # 每隔100次迭代，保存一次日志
        if step % 100 == 0:
            summary_str = session.run(merged_summary_op)
            summary_writer.add_summary(summary_str, step)

        # 每200次迭代输出一次损失 保存一次模型
        if step % get_config_int == 0:
            if step > 0:
                average_loss /= 2000
            average_loss = 0
            save_path = saver.save(sess, model_path, global_step=step)
            print("模型保存:{0}\t当前损失:{1}".format(model_path, average_loss))

        # 每step_num词隔迭代输出一次指定词语的最近邻居
        if step % 100 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = idx_2_vocab[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = idx_2_vocab[nearest[k]]
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
    labels = [idx_2_vocab[i] for i in xrange(plot_only)]
    embs_pic_path = get_config("word2vec", "embs_pic_path")
    plot_with_labels(low_dim_embs, labels, embs_pic_path)
except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
