#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

#####################################################################
##测试数据
##test data
#####################################################################

SCRIPT="./script"
DATA="./data_dir"
CONFIG="./configs"
sh $SCRIPT/build_dict.sh  $DATA/test/seg_qa
python $SCRIPT/word_2_vec.py $CONFIG/word_embedding_test.ini

#####################################################################
##全量qa训练数据
##whole train data
#####################################################################

