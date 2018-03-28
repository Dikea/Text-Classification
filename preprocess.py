#!/bin/env python
#-*- encoding: utf-8 -*-


import numpy as np
import codecs
from utils.nlp_util import NlpUtil
import config


def tokenize_corpus(corpus_fpath, save_fpath, is_train_data = True):

    def precess_line(line, is_train_data = True):
        try:
            line = line.strip()
            if is_train_data:
                line, flag = line.rsplit(',', 1)
            id_, text = line.split(',', 1)
            text = text.replace('|', ' ')
            text = text.replace('\t', ' ')
            text = '|'.join(['<s>'] + NlpUtil.tokenize(text, True) + ['</s>'])
            #text = '|'.join(NlpUtil.tokenize(text, True))
            return ('\t'.join([id_, text, flag]) + '\n' if is_train_data
                else '\t'.join([id_, text]) + '\n')
        except Exception as e:
            print ('line=%s, errmsg=%s', line, e)

    with codecs.open(corpus_fpath, 'r', 'utf-8') as in_f, \
        codecs.open(save_fpath, 'w', 'utf-8') as out_f:
        in_f.readline()
        for line in in_f:
            out_f.write(precess_line(line, is_train_data))
    print 'Tokenize done'


def _get_corpus():
    corpus = []
    for file_ in [config.train_fpath, config.predict_fpath]:
        with codecs.open(file_, 'r', 'utf-8') as in_f:
            corpus_tmp = [line.strip().split('\t')[1].split('|')
                          for line in in_f] 
            corpus.extend(corpus_tmp)
    print 'Get corpus done, length is %d' % len(corpus)
    return corpus


def build_emb_matrix(corpus):
    corpus_ = []
    _ = map(lambda x: corpus_.extend(x), corpus)
    word2id = NlpUtil.build_word2id(corpus_)
    word2vec = NlpUtil.load_word2vec(config.word2vec_fpath)
    emb_matrix = NlpUtil.build_emb_matrix(word2vec,
        config.embedding_size, word2id)
    np.save(config.emb_matrix_fpath, emb_matrix)
    with codecs.open(config.word2id_fpath, 'w', 'utf-8') as out_f:
        out_f.write('\n'.join(['%s\t%d' % (k, v) for k, v in word2id.iteritems()]))
    print 'Build emb_matrix done'


if __name__ == '__main__': 
    # Tokenize data
    tokenize_corpus(config.raw_train_fpath, config.train_fpath,
        is_train_data = True)
    tokenize_corpus(config.raw_predict_fpath, config.predict_fpath, 
        is_train_data = False)
    corpus = _get_corpus()

    # Train word2vec
    NlpUtil.train_word2vec(corpus, './model/word2vec')

    # Build emb matrix
    build_emb_matrix(corpus)
