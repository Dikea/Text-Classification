#!/bin/env python 
#-*- encoding: utf-8 -*-


import os
import time
import numpy as np
import tensorflow as tf
from utils import model_helper
import config


class ModelParas(object):
    embedding_size = config.embedding_size
    cell_num_units = 256
    num_layers = 1
    batch_size = 64
    cnn_dropout = 0.0
    rnn_dropout = 0.0 
    learning_rate = 0.01 
    decay = 0.99
    lrshrink = 5
    uniform_init_scale = 0.04
    clip_gradient_norm = 5.0
    filter_sizes = [3, 4, 5]
    l2_reg_lambda = 0.0
    max_pool_size = 4
    num_filters = 32
    epochs = 20 


class Model(object):
    

    def __init__(self, paras, sess, mode, emb_matrix):
        self.paras = paras
        self.sess = sess
        self.mode = mode

        # Model variable
        with tf.device('/cpu:0'):
            self.embeddings = tf.get_variable(
                name = 'embeddings',
                shape = emb_matrix.shape,
                dtype = tf.float32,
                initializer = tf.constant_initializer(emb_matrix))
        self.global_step = tf.get_variable( 
            name = 'global_step', 
            dtype = tf.int32,
            initializer = 1,
            trainable = False)

        self._build_graph()

    
    def _create_placeholder(self):
        self.lr = tf.placeholder(tf.float32, [], name = 'learning_rate')
        self.sents = tf.placeholder(tf.int32, [None, None], name = 'sents')
        with tf.device('/cpu:0'):
            self.emb_sents = tf.nn.embedding_lookup(
                self.embeddings, self.sents)
            # Expand dimension so meet input requirement of 2d-conv
            self.emb_expand = tf.expand_dims(self.emb_sents, -1)
        self.sent_lengths = tf.placeholder(tf.int32, [None], name = 'sent_lengths')
        self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
        self.labels = tf.placeholder(tf.int32, [None], name = 'labels')


    def _inference(self):
        # Convolution network
        with tf.name_scope('cnn'):

            # After conv and pooling, 
            max_length = tf.reduce_max(self.sent_lengths)
            div_value = tf.div(tf.cast(max_length, tf.float32), self.paras.max_pool_size)
            reduced_size = tf.cast(tf.ceil(div_value), tf.int32) 

            pooled_concat = []
            for i, filter_size in enumerate(self.paras.filter_sizes):
                with tf.name_scope('conv-pool-%s' % filter_size):
                    # Padding zero to keep conv output has same dimention as input
                    # shape is : [batch_size, sent_length, emb_size, channel]
                    num_prio = (filter_size - 1) // 2
                    num_post = (filter_size - 1) - num_prio
                    pad_prio = tf.concat([self.pad] * num_prio,1)
                    pad_post = tf.concat([self.pad] * num_post,1)
                    emb_pad = tf.concat([pad_prio, self.emb_expand, pad_post], 1)
                    # Prepare filter for conv
                    filter_ = tf.get_variable(
                        name = 'filter-%s' % filter_size,
                        shape = [filter_size, self.paras.embedding_size, 1, self.paras.num_filters])
                    # conv: [batch_size, sent_length, 1, num_filters]
                    conv = tf.nn.conv2d(
                        input = self.emb_pad,
                        filter = filter_,
                        strides = [1, 1, 1, 1],
                        padding = 'VALID',
                        name = 'conv')
                    # Bias
                    b = tf.get_variable(
                        name = 'bias-%s' % filter_size,
                        shape = [self.paras.num_filters])
                    h = tf.nn.relu(tf.nn.bias_add(conv, b))
                    # Max pooling over the outputs
                    pooled = tf.nn.max_pool(
                        value = h, 
                        ksize = [1, self.paras.max_pool_size, 1, 1],
                        trides = [1, self.paras.max_pool_size, 1, 1], 
                        padding ='SAME', 
                        name ='pool')
                    pooled = pooled.reshape(pooled, [-1, reduced_size, self.paras.num_filters])
                    pooled_concat.append(pooled)
            # pooled_concat: [batch_size, reduced_size, filter_sizes * num_filters]
            pooled_concat = tf.concat(pooled_concat, 2)
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN: 
                pooled_concat = tf.nn.dropout(pooled_concat, 1.0 - self.paras.cnn_dropout)

        # RNN network 
        with tf.name_scope('rnn'):
            cells_fw = model_helper.create_rnn_cell(
                'lstm', 
                self.paras.cell_num_units,
                self.paras.num_layers,
                self.paras.rnn_dropout,
                self.mode)
            cells_bw = model_helper.create_rnn_cell(
                'lstm', 
                self.paras.cell_num_units,
                self.paras.num_layers,
                self.paras.rnn_dropout,
                self.mode)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cells_fw, 
                cells_bw,
                inputs = pooled_concat,
                dtype = tf.float32)
            # states_fw: (batch_size, reduced_size, cell_size)
            states_fw, states_bw = outputs 
            concat_states = tf.concat([states_fw, states_bw], axis = 2)
            # sent_states: (batch_size, 2 * cell_size)
            self.sent_states = tf.reduce_max(concat_states, axis = 1)

        with tf.name_scope('classify'):
            hidden1 = tf.contrib.layers.fully_connected(
                inputs = self.sent_states,
                num_outputs = 512)
            hidden2 = tf.contrib.layers.fully_connected(
                inputs = hidden1,
                num_outputs = 5)
            self.predicts = tf.reduce_max(tf.contrib.layers.fully_connected(
                inputs = hidden2,
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


    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
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
        with tf.name_scope('summary') as scope:
            tf.summary.scalar('loss', self.loss) 
            tf.summary.scalar('accuracy', self.accuracy)


    def _build_graph(self):
        self._create_placeholder()
        self._inference()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()
        print 'Build graph done'


def test():
    sess = tf.Session()
    paras = ModelParas()

    emb_matrix = NlpUtil.build_emb_matrix()
    Model(paras, sess, tf.contrib.learn.ModeKeys.TRAIN)
    

if __name__ == '__main__':
    pass
