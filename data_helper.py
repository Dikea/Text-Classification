#!/bin/env python


import codecs
import numpy as np
from utils.nlp_util import NlpUtil
import config


class Helper(object):
    

    @classmethod
    def init(cls):
        pass


    @classmethod
    def sort_by_length(cls, sents, labels):
        len_array = np.array([len(s) for s in sents])
        len_perm = len_array.argsort()
        sents = sents[len_perm]
        labels = labels[len_perm]
        return sents, labels


    @classmethod
    def get_data(cls, is_train_data = True, partition = None, 
                 sort_flag = True, rand_seed = None):
        if rand_seed is not None:
            np.random.seed(rand_seed)

        word2id = {}
        with codecs.open(config.word2id_fpath, 'r', 'utf-8') as in_f:
            for line in in_f:
                word, id_ = line.rstrip().split('\t')
                word2id[word] = int(id_)

        def split_text(text):
            ret = [word2id[w] for w in text.split('|') if w in word2id]
            return ret

        if is_train_data:
            # Return data for training
            if partition is None:
                partition = [0.8, 0.1, 0.1]
            partition = [0.0] + [sum(partition[:id_+1]) for id_ in range(3)]
            with codecs.open(config.train_fpath, 'r', 'utf-8') as in_f:
                train_corpus = [line.strip().split('\t') for line in in_f]
                train_data = [split_text(item[1]) for item in train_corpus]
                labels = np.array([int(item[2]) for item in train_corpus], 
                    dtype = np.int32) - 1
            train_length = len(train_data)
            perm = np.random.permutation(train_length)
            train_data = np.array(train_data)[perm]
            labels = labels[perm]
            train, dev, test = {}, {}, {}
            data_type = ['train', 'dev', 'test']
            part = np.array(partition) * train_length
            part = part.astype(np.int32)    
            for id_, type_ in enumerate(data_type):
                sents_ = train_data[part[id_] : part[id_+1]]
                labels_ = labels[part[id_] : part[id_+1]]
                if sort_flag is True:
                    sents_, labels_ = cls.sort_by_length(sents_, labels_)
                eval(type_)['sents'] = sents_ 
                eval(type_)['labels'] = labels_ 
            # print len(train['sents']), len(dev['sents']), len(test['sents'])
            # print '|'.join(map(str, test['sents'][-1])), test['labels'][-1]
            return train, dev, test
        else:
            # Return data for prediction
            with codecs.open(config.predict_fpath, 'r', 'utf-8') as in_f:
                predict_corpus = [line.strip().split('\t') for line in in_f]
                predict_ids = np.array([item[0] for item in predict_corpus])
                predict = np.array([split_text(item[1]) for item in predict_corpus])
                if sort_flag:
                    predict, predict_ids = cls.sort_by_length(predict, predict_ids)
            return predict_ids, predict


    @classmethod
    def get_batch(cls, batch, sequence_length = None):
        if sequence_length:
            lengths = np.array([len(x[:sequence_length]) for x in batch])
        else:
            lengths = np.array([len(x) for x in batch])
        max_len = np.max(lengths)
        batch_len = len(batch)
        embed = np.zeros((batch_len, max_len), np.int32)
        for i in xrange(batch_len):
            for j in xrange(lengths[i]):
                embed[i, j] = batch[i][j] 
        return embed, lengths


    @classmethod
    def get_emb_matrix(cls):
        emb_matrix = np.load(config.emb_matrix_fpath)
        print 'Load embedding matrix success' 
        return emb_matrix


def test():
    train, dev, test = Helper.get_data(is_train_data = True, 
                                       sort_flag = False, 
                                       rand_seed = 1234)

    print train['sents'][:3]
    batch = Helper.get_batch(train['sents'][:3])
    print batch
    Helper.get_emb_matrix()


if __name__ == '__main__':
    test()

