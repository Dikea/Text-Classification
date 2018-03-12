#!/bin/env python 
#-*- encoding: utf-8 -*-


import os
import time
import tensorflow as tf
from utils import model_helper
import config


class ModelParas(object):
    embedding_size = config.embedding_size
    batch_size = 64
    sequence_length = 70
    learning_rate = 0.01 
    decay = 0.99
    lrshrink = 5
    uniform_init_scale = 0.04
    clip_gradient_norm = 5.0
    l2_reg_lambda = 0.0
    nclasses = 5
    epochs = 20 

    # RNN
    cell_num_units = 256
    num_layers = 1
    rnn_dropout = 0.0 

    # CNN
    filter_sizes = [3, 4, 5]
    num_filters = 32
    cnn_dropout = 0.0


class Model(object):
    

    def __init__(self, paras, sess, mode, emb_matrix):
        self.paras = paras
        self.sess = sess
        self.mode = mode
        self.emb_matrix = emb_matrix
        self._build_graph()


    def _create_placeholder(self):
        self.lr = tf.placeholder(tf.float32, [], name = 'learning_rate')
        self.sents = tf.placeholder(tf.int32, [None, None], name = 'sents')
        self.sent_lengths = tf.placeholder(tf.int32, [None], name = 'sent_lengths')
        self.labels = tf.placeholder(tf.int32, [None], name = 'labels')


    def _create_variable(self):
        with tf.device('/cpu:0'):
            self.embeddings = tf.get_variable(
                name = 'embeddings',
                shape = self.emb_matrix.shape,
                dtype = tf.float32,
                initializer = tf.constant_initializer(self.emb_matrix))
        self.global_step = tf.get_variable( 
            name = 'global_step', 
            dtype = tf.int32,
            initializer = 1,
            trainable = False)
        self.num_filters_total = self.paras.num_filters * len(self.paras.filter_sizes)
        self.w_projection = tf.get_variable(
            name = 'w_projection', 
            shape = [self.num_filters_total, self.paras.nclasses])
        self.b_projection = tf.get_variable(
            name = 'b_projection', 
            shape = [self.paras.nclasses])
        self.l2_loss = tf.constant(0.0)

    
    def _inference(self):
        paras = self.paras
        with tf.device('/cpu:0'):
            self.emb_sents = tf.nn.embedding_lookup(
                self.embeddings, self.sents)

        # RNN network
        with tf.name_scope('RNN'):
            cells_fw = model_helper.create_rnn_cell(
                'lstm', 
                paras.cell_num_units,
                paras.num_layers,
                paras.rnn_dropout,
                self.mode)
            cells_bw = model_helper.create_rnn_cell(
                'lstm', 
                paras.cell_num_units,
                paras.num_layers,
                paras.rnn_dropout,
                self.mode)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cells_fw, 
                cells_bw,
                inputs = self.emb_sents,
                sequence_length = self.sent_lengths,
                dtype = tf.float32)
            # states_fw: (batch_size, sent_len, cell_size)
            states_fw, states_bw = outputs 
            # concat_states: (batch_size, sent_len, cell_size * 2)
            concat_states = tf.concat([states_fw, states_bw], axis = 2)
            # rnn_states_expand: (batch_size, sent_len, cell_size * 2, 1)
            self.rnn_states_expand = tf.expand_dims(concat_states, -1)

        # CNN network
        with tf.name_scope('CNN'):
            pooled_concat = []
            for filter_size in paras.filter_sizes:
                with tf.name_scope('conv-pool-%s' % filter_size):
                    # filter: (shape)
                    filter_ = tf.get_variable(
                        name = 'filter-%s' % filter_size,
                        shape = [filter_size, paras.cell_num_units * 2, 1, paras.num_filters])
                    # conv: (batch_size, sequence_length - filter + 1, 1, num_filters)
                    conv = tf.nn.conv2d(
                        input = self.rnn_states_expand,
                        filter = filter_,
                        strides = [1, 1, 1, 1],
                        padding = 'VALID',
                        name = 'conv')
                    # bias: (num_filters, 1)
                    b = tf.get_variable(
                        name = 'bias-%s' % filter_size,
                        shape = [paras.num_filters])
                    h = tf.nn.relu(tf.nn.bias_add(conv, b))
                    # pooled: (batch_size, 1, 1, num_filters)
                    pooled = tf.nn.max_pool(
                        value = h, 
                        ksize = [1, paras.sequence_length - filter_size + 1, 1, 1],
                        strides = [1, 1, 1, 1], 
                        padding ='VALID', 
                        name ='pool')
                    pooled_concat.append(pooled)
            # h_pool: (batch_size, 1, 1, num_filters_total)
            h_pool = tf.concat(pooled_concat, 3)
            # h_pool_flat: (batch_size, num_filters_total)
            self.h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])
            # dropout 
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                self.h_pool_flat = tf.nn.dropout(self.h_pool_flat, 1.0 - paras.cnn_dropout)

        with tf.name_scope('classify'):
            # logits: (batch_size, n_classes)
            logits = tf.nn.xw_plus_b(self.h_pool_flat, w_projection, b_projection, 'logits')
            # predicts: (batch_size, 1)
            self.predicts = tf.reduce_max(tf.contrib.layers.fully_connected(
                inputs = logits,
                activation_fn = None,
                num_outputs = 1), axis = 1)
            self.mse = tf.reduce_mean(tf.cast(
                tf.squared_difference(
                    self.labels, 
                    tf.cast(tf.round(self.predicts), tf.int32)),
                tf.float32))

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.labels, 
                tf.cast(tf.round(self.predicts), tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(
                correct_prediction, tf.float32))


    def _create_loss(self):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    labels = tf.cast(self.labels, tf.float32),
                    predictions = self.predicts)) 
            # Add l2 loss reg
            l2_loss += tf.nn.l2_loss(w_projection)
            l2_loss += tf.nn.l2_loss(b_projection)
            self.loss += l2_loss * self.paras.l2_reg_lambda


    def _create_optimizer(self):
        self.optimizer = tf.contrib.layers.optimize_loss(
            loss = self.loss, 
            global_step = self.global_step, 
            learning_rate = self.lr, 
            optimizer = 'SGD', 
            clip_gradients = self.paras.clip_gradient_norm) 


    def _create_summary(self):
        log_path = os.path.join(config.model_path, 'tensorboard')
        self.train_writer = tf.summary.FileWriter(
            os.path.join(log_path, 'train'), self.sess.graph)
        self.test_writer = tf.summary.FileWriter(
            os.path.join(log_path, 'test'), self.sess.graph)
        with tf.name_scope('summaries') as scope:
            tf.summary.scalar('loss', self.loss) 
            tf.summary.scalar('accuracy', self.accuracy)


    def _build_graph(self):
        self._create_placeholder()
        self._create_variable()
        self._inference()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()
        print 'Build graph done'


def test():
    from data_helper import Helper
    sess = tf.Session()
    paras = ModelParas()
    emb_matrix = Helper.get_emb_matrix()
    Model(paras, sess, tf.contrib.learn.ModeKeys.TRAIN, emb_matrix)
    

if __name__ == '__main__':
    test()
