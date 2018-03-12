#!/bin/env python
#-*- encoding: utf-8 -*-


# Data
raw_train_fpath = './data/train_first.csv'
raw_predict_fpath = './data/predict_first.csv'
train_fpath = './data/train.txt'
predict_fpath = './data/predict.txt'


# Word2vec
embedding_size = 300 
word2vec_fpath = './model/word2vec/w2v_win1_d%d.model' % embedding_size
emb_matrix_fpath = './model/word2vec/emb_matrix_d%d.npy' % embedding_size
word2id_fpath = './model/word2vec/word2id.txt'


# Model path
model_path = './model/m0'


# Result path
result_path = './data/result.csv'
