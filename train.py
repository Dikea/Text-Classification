#!/bin/env python
#-*- encoding: utf-8 -*-


import os
import time
import math
import codecs
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from rnn_model import Model, ModelParas
from data_helper import Helper
from predict import predict, fuse_result
from utils.log import logger
import config


tf.flags.DEFINE_string('model', 'rnn', 'select model, default is rnn')
tf.flags.DEFINE_string('mode', 'single', 'single, multi or kfold, default is single') 
flags = tf.flags.FLAGS


def run_epoch(model, input_data):
    start_time = time.time()
    paras = model.paras
    average_loss, average_acc, average_mse = 0.0, 0.0, 0.0
    sents, labels = input_data['sents'], input_data['labels']
    data_length = len(sents)
    if data_length == 0: 
        return None 
    steps = int(math.ceil(data_length * 1.0 / paras.batch_size))

    for step in xrange(steps):
        begin = step * paras.batch_size
        end = (step + 1) * paras.batch_size
        batch_sents, batch_lengths = Helper.get_batch(
            sents[begin: end], paras.sequence_length)
        batch_labels = labels[begin: end]
        feed_dict = {
            model.sents: batch_sents,
            model.sent_lengths: batch_lengths,
            model.labels: batch_labels.T,
            model.lr: paras.learning_rate}
        if flags.model == 'cnn_rnn':
            feed_dict[model.pad] = np.zeros((
                len(labels[begin: end]), 1, paras.embedding_size, 1))
        fetches = {
            'b_loss': model.loss,
            'b_acc': model.accuracy,
            'global_step': model.global_step,
            'b_mse': model.mse,
        }
        if model.mode == tf.contrib.learn.ModeKeys.TRAIN:
            fetches['optimizer'] = model.optimizer
        vals = model.sess.run(fetches, feed_dict)
        b_loss, b_acc, b_mse, global_step = (
            vals['b_loss'], vals['b_acc'], 
            vals['b_mse'], vals['global_step'])
        b_score = 1.0 / (1.0 + np.sqrt(b_mse))
        average_loss += b_loss 
        average_acc += b_acc 
        average_mse += b_mse
        if (model.mode == tf.contrib.learn.ModeKeys.TRAIN and global_step % 10 == 0):
            logger.debug('step=%d, b_loss=%.4f, b_acc=%.4f, b_mse=%.4f, b_score=%.4f', 
                global_step, b_loss, b_acc, b_mse, b_score)

    average_loss /= steps
    average_acc /= steps
    average_mse /= steps
    rmse_score = 1.0 / (1.0 + np.sqrt(average_mse))
    logger.debug('average_loss=%.4f, average_acc=%.4f, average_mse=%.4f, rmse_score=%.4f', 
        average_loss, average_acc, average_mse, rmse_score)
    return rmse_score, global_step


def train(train_data, valid_data, test_data, emb_matrix):
    """Train the model"""
    start_time = time.time()
    paras = ModelParas()
    tf.reset_default_graph()
    sess = tf.Session()
    # Init initialzer
    uniform_initializer = tf.random_uniform_initializer(
        minval = -paras.uniform_init_scale, 
        maxval = paras.uniform_init_scale)
    # Define model for train and evaluate
    with tf.name_scope('train'):
        with tf.variable_scope('Model', reuse = None, 
                                initializer = uniform_initializer):
            model_train = Model(paras, 
                                sess, 
                                tf.contrib.learn.ModeKeys.TRAIN,
                                emb_matrix)
    with tf.name_scope('valid'):
        with tf.variable_scope('Model', reuse = True, 
                                initializer = uniform_initializer):
            model_eval = Model(paras, 
                               sess, 
                               tf.contrib.learn.ModeKeys.EVAL,
                               emb_matrix)
    # Model Train
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    best_score = -np.inf
    saver = tf.train.Saver()
    save_path = os.path.join(config.model_path, 'model/model.ckpt')
    for epoch in xrange(paras.epochs):
        logger.debug('>>> Epoch %d, learning_rate=%.4f', 
                     epoch, paras.learning_rate) 
        run_epoch(model_train, train_data) 
        logger.debug('>>> Running Valid')
        score, global_step = run_epoch(model_eval, valid_data)
        if score > best_score:
            best_score = score
            saver.save(sess, save_path)
            logger.debug('Score improved, save model to %s', save_path)
        else:
            saver.restore(sess, save_path)
            logger.debug('Score not improved, load previous best model')
        logger.debug('Epoch %d done, time=%.4f minutes', 
                     epoch, (time.time() - start_time) / 60)
    logger.debug('>>> Running Test')
    run_epoch(model_eval, test_data)
    del model_train
    del model_eval
    logger.debug('Predict result')
    predict(save_path = os.path.join(config.model_path, 
        'result_%f' % best_score))


def tmp_predict(model, save_path):
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
                out_f.write('%s,%f\n' % (id_, val))
    del predict_ids, predict
    print 'Predict done'



def main(_):
    start_time = time.time()
    logger.info('Train begin...')
    emb_matrix = Helper.get_emb_matrix()
    if flags.mode == 'single':
        train_data, valid_data, test_data = Helper.get_data(
            is_train_data = True, partition = [0.8, 0.2], rand_seed = 666) 
        train(train_data, valid_data, test_data, emb_matrix)
    elif flags.mode == 'multi':
        for i in range(10):
            print '>>> Multi %d' % i
            train_data, valid_data, test_data = Helper.get_data(
                is_train_data = True, partition = [0.8, 0.2], rand_seed = None) 
            train(train_data, valid_data, test_data, emb_matrix)
        fuse_result()
    elif flags.mode == 'kfold':
        data_, _, _  = Helper.get_data(
            is_train_data = True, partition = [1.0], sort_flag = False) 
        sents, labels = data_['sents'], data_['labels']
        kf = KFold(n_splits = 10, shuffle = True, random_state = 123)
        train_data, test_data = {}, {}
        cnt = 1
        for train_index, test_index in kf.split(sents):
            print '>>> KFold %d' % cnt
            cnt += 1
            train_data['sents'] = sents[train_index] 
            train_data['labels'] = labels[train_index]
            test_data['sents'] = sents[test_index]
            test_data['labels'] = labels[test_index]
            train_data['sents'], train_data['labels'] = Helper.sort_by_length(
                train_data['sents'], train_data['labels'])
            test_data['sents'], test_data['labels'] = Helper.sort_by_length(
                test_data['sents'], test_data['labels'])
            train(train_data, test_data, _, emb_matrix)
        fuse_result()
    else:
        raise ValueError('Train mode must be `single | multi | kfold` !') 
    logger.info('Train done, time=%.4f hours' % ((time.time() - start_time) / 3600))


if __name__ == '__main__':
    log_path = './log/train.log'
    if os.path.exists(log_path):
        os.remove(log_path)
    logger.start(log_path, name = __name__)
    model_path = config.model_path
    if tf.gfile.Exists(model_path):
        tf.gfile.DeleteRecursively(model_path)
        logger.debug('Remove old model folder.')
    tf.app.run()
