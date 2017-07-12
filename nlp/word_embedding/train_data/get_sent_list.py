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
    drop_freq=5
    del_threadhold=0.8
    raw_words_file=sys.argv[1]
    stop_word_file=sys.argv[2]
    raw_qa_file=sys.argv[3]
    #pickle_file=sys.argv[4]

    stop_words = set(open(stop_word_file).readlines())
    word_counts = {line.split("\t")[0]:int(line.split("\t")[1])  for line in open(raw_words_file).readlines()}
            
    # 计算总词频
    total_count = sum(word_counts.values())
    #print("total words:{}".format(total_count))
    word_freq = {w:float( c) / total_count for w, c in word_counts.items()}
    prob_drop = {w: 1 - np.sqrt(1e-5 / f) for w, f in word_freq.items()}
    # 将低频和停用词都剔除成为训练数据，被剔除的使用UNK做平滑
    train_words = map(lambda w: "UNK" if((word_counts[w] <= drop_freq) or (w in stop_words) or (prob_drop[w]>=del_threadhold)) else w, word_counts.keys())
    vocab_2_idx ={w:i for i,w in enumerate(train_words)}
    print("unique words:{}".format(len(vocab_2_idx)))
    #得到qa向量化以后的数据
    line = raw_qa.readline()
    while line:
        qa_sents = line.split(qa_split)
        if len(qa_sents)==2:
            q_list = map(lambda w: vocab_2_idx["UNK"] if w not in vocab_2_idx else vocab_2_idx[w],qa_sents[0].split(word_split))
            a_list = map(lambda w: vocab_2_idx["UNK"] if w not in vocab_2_idx else vocab_2_idx[w],qa_sents[1].split(word_split))
	    print("".join([str(idx) for idx in q_list]))
	    print("".join([str(idx) for idx in a_list]))
	line = raw_qa.readline()
    raw_qa.close()
'''
    print("total words:{}".format(len(train_words)))
    print("unique words:{}".format(len(vocab_2_idx)))
    print("Dump file is:{}".format(pickle_file))
'''

if len(sys.argv)!=4:
	print("useage: python build_dict.py dict_word_file stop_word_file raw_qa_file ")
	exit(1)
build_dict()

