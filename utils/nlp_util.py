#!/bin/env python
#-*- encoding: utf-8 -*-


import os
import time
import jieba
import codecs
import collections
import numpy as np
from gensim import models
import config


class NlpUtil(object):
    

    @classmethod
    def tokenize(cls, text, filter_stop_word = False):
        if not isinstance(text, unicode):
            return [str(text)]
        tokens = jieba.lcut(text)
        if filter_stop_word:
            stop_word_set = config.stop_word_set
            tokens = filter(lambda w: w not in stop_word_set, tokens)
        return tokens


    @classmethod
    def train_word2vec(cls, corpus, wv_fpath = ''):
        time_s = time.time()
        vec_size = 300
        win_size = 1
        print ('begin to train model...')
        w2v_model = models.word2vec.Word2Vec(corpus,
                                             size = vec_size,
                                             window = win_size,
                                             min_count = 2,
                                             workers = 4,
                                             sg = 1,
                                             negative = 15,
                                             iter = 7)
        w2v_model.train(corpus, total_examples = len(corpus), epochs = w2v_model.iter)
        save_fpath = os.path.join(wv_fpath,
            'w2v_win%s_d%s.model' % (win_size, vec_size))
        w2v_model.save(save_fpath)
        print ('save model success, model_path=%s, time=%.4f sec.' 
                % (save_fpath, time.time() - time_s))


    @classmethod 
    def load_word2vec(cls, w2v_fpath):
        w2v_model = models.word2vec.Word2Vec.load(w2v_fpath)
        print 'load word2vec success'
        wv = w2v_model.wv
        del w2v_model
        return wv


    @classmethod
    def build_word2id(cls, corpus): 
        """Convert corpus from word to id
        Args:
            corpus: a list of all words   

        Returns:
            word_to_id: a dict of word to id
        """
        counter = collections.Counter(corpus)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        word2id = dict(zip(words, range(1, len(words) + 1)))
        return word2id

    
    @classmethod
    def build_emb_matrix(cls, word2vec, emb_size, word2id, 
                         init_scale = 0.25, norm_flag = False):
        vocab_size = len(word2id) 
        emb_matrix = np.zeros((vocab_size + 1, emb_size), np.float32)
        for w, id_ in word2id.iteritems():
            if w in word2vec:
                emb_matrix[id_] = word2vec[w] 
            else:
                emb_matrix[id_] = np.random.uniform(
                    -init_scale, init_scale, emb_size)   
        return emb_matrix
     

def test():
    # Test tokenize
    print '|'.join(NlpUtil.tokenize(u'天气很好')).encode('utf-8')
    
    '''
    # Test word2vec
    wv = NlpUtil.load_word2vec('./model/word2vec/w2v_win1_d128.model')
    print wv[u'天气']
    print '|'.join([x[0] for x in wv.most_similar(positive = [u'天气'])]).encode('utf-8')
    '''
    


if __name__ == '__main__':
    test()
