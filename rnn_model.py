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
    sequence_length = None
    cell_num_units = 512
    num_layers = 1
    batch_size = 64
    dropout = 0.0 
    learning_rate = 0.01 
    decay = 0.99
    lrshrink = 5
    uniform_init_scale = 0.04
    clip_gradient_norm = 5.0
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
        self.sent_lengths = tf.placeholder(tf.int32, [None], name = 'sent_lengths')
        self.labels = tf.placeholder(tf.int32, [None], name = 'labels')


    def _inference(self):
        with tf.variable_scope('encoder') as varscope:
            cells_fw = model_helper.create_rnn_cell(
                'lstm', 
                self.paras.cell_num_units,
                self.paras.num_layers,
                self.paras.dropout,
                self.mode)
            cells_bw = model_helper.create_rnn_cell(
                'lstm', 
                self.paras.cell_num_units,
                self.paras.num_layers,
                self.paras.dropout,
                self.mode)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cells_fw, 
                cells_bw,
                inputs = self.emb_sents,
                sequence_length = self.sent_lengths,
                dtype = tf.float32,
                scope = varscope)
            # states_fw: (batch_size, sent_len, cell_size)
            states_fw, states_bw = outputs 
            concat_states = tf.concat([states_fw, states_bw], axis = 2)
            # sent_states: (batch_size, 2 * cell_size)
            self.sent_states = tf.reduce_max(concat_states, axis = 1)

        with tf.variable_scope('classify_layer') as varscope:
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

        with tf.variable_scope('accuracy') as varscope:
            correct_prediction = tf.equal(self.labels, 
                tf.cast(tf.round(self.predicts), tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(
                correct_prediction, tf.float32))


    def _create_loss(self):
        with tf.variable_scope('loss') as varscope:
            self.loss = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    labels = tf.cast(self.labels, tf.float32),
                    predictions = self.predicts)) 


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
