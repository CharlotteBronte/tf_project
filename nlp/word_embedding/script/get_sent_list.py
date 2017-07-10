#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@desc:  通过words表得到qa句子的id表示，返回结果存储在$1.pickle中
@time:  2017/07/09 16:25
@author: liuluxin(0_0mirror@sina.com)
@param: $1:words词表（不去重）$2:stopwords词表 $2:qu数据文件
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import pickle
import collections
from collections import Counter
from tensorflow.contrib.tensorboard.plugins import projector
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np

qa_split=""
word_split=""


def build_dict():
    word_freq=1
    del_threadhold=1.0
    raw_words_file=sys.argv[1]
    stop_word_file=sys.argv[2]
    raw_qa_file=sys.argv[3]
    pickle_file=sys.argv[4]

    stop_words = set(open(stop_word_file).readlines())
    raw_words = [w.strip() for w in open(raw_words_file).readlines()]
    print("Raw words:{}".format(len(raw_words)))
    word_counts = Counter(raw_words)
    # 计算总词频
    total_count = len(raw_words)
    word_freq = {w: c / total_count for w, c in word_counts.items()}
    prob_drop = {w: 1 - np.sqrt(1e-5 / f) for w, f in word_freq.items()}
    # 将低频和停用词都剔除成为训练数据，被剔除的使用UNK做平滑
    train_words = map(lambda w: "UNK" if((word_counts[w] <= word_freq) or (w in stop_words) or (prob_drop[w]>=del_threadhold)) else w, raw_words)
    vocab_2_idx ={w:i for i,w in enumerate(set(train_words))}
    #得到qa向量化以后的数据
    q_list=[[]]
    a_list=[[]]
    raw_qa = open(raw_qa_file)
    line = raw_qa.readline()
    while line:
        qa_sents = line.split(qa_split)
        if len(qa_sents)==2:
            q_list.append(map(lambda w: vocab_2_idx["UNK"] if w not in vocab_2_idx else vocab_2_idx[w],qa_sents[0].split(word_split)))
            a_list.append(map(lambda w: vocab_2_idx["UNK"] if w not in vocab_2_idx else vocab_2_idx[w],qa_sents[1].split(word_split)))
	line = raw_qa.readline()
    raw_qa.close()

    #将词典的和qa数据dump到文件中
    pick= open(pickle_file, 'wb')
    pickle.dump(vocab_2_idx, pick)
    pickle.dump(q_list, pick)
    pickle.dump(a_list, pick)
    pick.close()
    print("Total words:{}".format(len(train_words)))
    print("Unique words:{}".format(len(vocab_2_idx)))
    print("Dump file is:{}".format(pickle_file))

if len(sys.argv)!=5:
	print("useage: python build_dict.py dict_word_file stop_word_file raw_qa_file pickle_file")
	exit(1)
build_dict()

