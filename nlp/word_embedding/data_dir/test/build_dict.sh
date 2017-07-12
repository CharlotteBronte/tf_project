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
if [ $# != 2 ];then
echo "useage sh build_dict.sh seg_qa_file seg_qa_id_file" 
exit 1
fi
raw_file=$1
out_file=$2
rm -rf tmp_dir 
mkdir tmp_dir 
all_size=`wc -l $1|cut -d' ' -f1`
echo $all_size
batch_size=$(($all_size/32))
python get_sent_list.py dict $1.words stop_words $1.pickle
for idx in {0..31}
do
idx_plus=$(($idx+1))
head -n $(($idx_plus*$batch_size))   $1| tail -n $(($idx*$batch_size)) > tmp_dir/$2_test.$idx
python get_sent_list.py list $1.pickle tmp_dir/$2_test.$idx  >tmp_dir/$2.$idx &
done
head -n $all_size   $1| tail -n $((31*$batch_size))> tmp_dir/$2_test.$idx &
