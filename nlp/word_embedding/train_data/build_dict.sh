#!/bin/bash
#####################################################################
## Copyright (c) 2017 roobot.com, Inc. All Rights Reserved
## @file:  build_data.sh
## @desc:  build的词典
## @author: liuluxin(0_0mirror@sina.com)
## @date: 2017/07/10  14:39
## @version: 1.0.0.0
## @param:  
## @return:
#####################################################################
raw_file=$1
awk '{printf $0""}' $raw_file |sed 's///g' | awk -F"" '{for(i=1;i<=NF;i++){print $i}}'|sed 's/\s\+//g' |sort  > $1.words 
#pyhton get_sent_list.py $1  $1.sents
