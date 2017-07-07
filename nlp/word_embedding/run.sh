#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
python word_2_vec.py word_embedding_test.ini
