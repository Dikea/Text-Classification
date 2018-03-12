#!/bin/env python
#-*- encoding: utf-8 -*-


import os
import time
import math
import numpy as np
from collections import defaultdict, Counter
import codecs
import tensorflow as tf
from rnn_model import Model, ModelParas
from data_helper import Helper
from utils.log import logger
import config


def load_model(mode):
    tf.reset_default_graph()
    paras = ModelParas()
    sess = tf.Session()
    save_path = os.path.join(config.model_path, 'model/model.ckpt')
    emb_matrix = Helper.get_emb_matrix()
    with tf.variable_scope('Model'):
        model = Model(paras, sess, mode, emb_matrix)
    saver = tf.train.Saver()
    saver.restore(model.sess, save_path)
    return model


def predict(save_path):
    model = load_model(mode = tf.contrib.learn.ModeKeys.EVAL)
    predict_ids, predict = Helper.get_data(is_train_data = False)
    batch_size = model.paras.batch_size
    steps = int(math.ceil(len(predict_ids) * 1.0 / batch_size))
    with codecs.open(save_path, 'w', 'utf-8') as out_f:
        for step in xrange(steps):
            begin = step * batch_size
            end = (step + 1) * batch_size
            ids = predict_ids[begin: end]
            batch_sents, batch_lengths = Helper.get_batch(
                predict[begin: end], model.paras.sequence_length)
            feed_dict = {
                model.sents: batch_sents,
                model.sent_lengths: batch_lengths}
            res = model.sess.run(model.predicts, feed_dict)
            ids = ids.tolist()
            res = res.tolist()
            msgs = predict[begin: end].tolist()
            for id_, val, msg in zip(ids, res, msgs):
                out_f.write('%s,%f\n' % (id_, val + 1))
    del model, predict_ids, predict
    print 'Predict done'


def fine_tune_result():
    ratio = np.array([0.00587, 0.00973, 0.09389, 0.28954, 0.60097], np.float32)
    part = np.array([np.sum(ratio[:i]) for i in range(6)]) * 30000
    part[-1] = 30000
    part = part.astype(np.int32) 
    print part
    with codecs.open(config.result_path, 'r', 'utf-8') as in_f, \
        codecs.open('fine_tune.csv', 'w', 'utf-8') as out_f:
        id_score_list = []
        for line in in_f:
            id_, score = line.rstrip().split(',')
            id_score_list.append((id_, float(score)))
        id_score_list.sort(key = lambda x: x[1])
        for index, item in enumerate(id_score_list):
            for i in range(5):
                if part[i] <= index < part[i + 1]:
                    out_f.write('%s,%d\n' % (item[0], i + 1)) 
                    break
    print 'Fine tune result done'
   

def _get_vote_value(array):
    array = [int(np.round(x)) for x in array]
    cnt_dict = Counter(array) 
    max_v = max([v for k, v in cnt_dict.items()])
    for k, v in cnt_dict.items()[::-1]:
        if v == max_v:
            return k


def _get_mean_value(array):
    return np.mean(array) 
    

def fuse_result(fuse_mode = 'mean'):
    id2result = defaultdict(list)
    total_score = 0.0 
    file_count = 0
    for file_ in os.listdir(config.model_path):
        if not file_.startswith('result'):
            continue
        file_count += 1
        file_ = os.path.join(config.model_path, file_)
        with codecs.open(file_, 'r', 'utf-8') as in_f:
            score = float(file_.split('_')[1])
            total_score += score
            for line in in_f:
                id_, kind_ = line.strip().split(',')
                tuple_ = (float(kind_), score)
                id2result[id_].append(tuple_)
    with codecs.open(config.result_path, 'w', 'utf-8') as out_f:
        for id_, list_ in id2result.iteritems():
            array = [kind_ for kind_, score_ in list_]
            if fuse_mode == 'mean':
                fuse_kind = _get_mean_value(array)
            else:
                fuse_kind = _get_vote_value(array)
            out_f.write('%s,%f\n' % (id_, fuse_kind))
    print id2result['16866b2f-c7e5-319d-b47b-cc9317812bc9'] 
    print 'Fuse result done'


if __name__ == '__main__':
    #predict(config.result_path)
    fuse_result()
