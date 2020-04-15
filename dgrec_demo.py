#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import roc_auc_score
import pickle, argparse, datetime, os, sys, itertools, time, scipy, math

parser = argparse.ArgumentParser()
parser.add_argument('-MAX_TIME_STEP', type=int, default=10)
parser.add_argument('-batch_size', type=int, default=512)
parser.add_argument('-user_num', type=int, default=512)
parser.add_argument('-item_num', type=int, default=512)
parser.add_argument('-emb_dim', type=int, default=128)
parser.add_argument('-friends_num', type=int, default=12)
parser.add_argument('-layer_num', type=int, default=2)
parser.add_argument('-lr', type=int, default=1e-3)
args = parser.parse_args()


class DGRec():
    def __init__(self):
        stdv = 1.0 / math.sqrt(args.emb_dim)
        self.initializer = tf.random_uniform_initializer(-stdv, stdv)
        self.build_graph()

    def build_graph(self):
        self.target_user = tf.placeholder(tf.int32, [args.batch_size], 'target_user')
        self.sess_seq = tf.placeholder(tf.int32, [args.batch_size, None], 'sess_seq')
        self.target_item = tf.placeholder(tf.int32, [args.batch_size], 'target_item')
        self.click_label = tf.placeholder(tf.int32, [args.batch_size], 'click_label')
        self.friends = tf.placeholder(tf.int32, [args.batch_size, args.friends_num], 'friends')
        self.friends_sess_seq = tf.placeholder(tf.int32, [args.batch_size, args.friends_num, args.MAX_TIME_STEP],
                                               'friends_sess_seq')
        # lookup dict
        self.user_embed_dict = tf.get_variable('user_embed_dict', [args.user_num, args.emb_dim],
                                               tf.float32, self.initializer)
        self.item_embed_dict = tf.get_variable('item_embed_dict', [args.item_num, args.emb_dim],
                                               tf.float32, self.initializer)
        #
        target_user = tf.nn.embedding_lookup(self.user_embed_dict, self.target_user)
        sess_seq = tf.nn.embedding_lookup(self.item_embed_dict, self.sess_seq)
        target_item = tf.nn.embedding_lookup(self.item_embed_dict, self.target_item)
        # lstm
        cell = tf.nn.rnn_cell.LSTMCell(args.emb_dim, activation=lambda x: tf.tanh(x), name='lstm')
        zero_state = cell.zero_state(args.batch_size, tf.float32)
        sess_len = tf.count_nonzero(self.sess_seq, 1)
        # h_n-公式(1)
        _, state = tf.nn.dynamic_rnn(cell, sess_seq, sequence_length=sess_len, initial_state=zero_state)
        h_n = state[1]
        # s_k_s - 公式(3)
        all_friends = tf.reshape(self.friends_sess_seq, [-1, args.MAX_TIME_STEP])
        friends_sess_seq = tf.nn.embedding_lookup(self.item_embed_dict, all_friends)
        zero_state_f = cell.zero_state(args.batch_size * args.friends_num, tf.float32)
        sess_len_f = tf.count_nonzero(all_friends, 1)
        # lstm复用
        _, state_f = tf.nn.dynamic_rnn(cell, friends_sess_seq, sequence_length=sess_len_f,
                                       initial_state=zero_state_f)
        s_k_s = state_f[1]  # tf.reshape(state_f[1],[args.batch_size,args.friends_num,-1])
        # s_k_l - 公式(4)
        friends = tf.nn.embedding_lookup(self.user_embed_dict, tf.reshape(self.friends, [-1]))
        # s_k - 公式(5)
        s_k = tf.layers.dense(inputs=tf.concat([s_k_s, friends], 1), units=args.emb_dim,
                              activation=tf.nn.relu, use_bias=False)
        s_k = tf.reshape(s_k, [args.batch_size, args.friends_num, -1])
        # dynamic feature graph
        h_u = h_n
        for l_idx in range(args.layer_num):
            # similarity - 公式(6)
            a_uk = tf.nn.softmax(tf.matmul(s_k, tf.reshape(h_u, [args.batch_size, args.emb_dim, 1])), 1)
            # combine friends interest - 公式(7)
            h_u_ = tf.reduce_sum(a_uk * s_k, 1)
            h_u = tf.layers.dense(inputs=h_u_, units=args.emb_dim,
                                  activation=tf.nn.relu, use_bias=False, name='dense_l%d' % l_idx)
        # final user interest - 公式(8)
        h_n = tf.layers.dense(inputs=tf.concat([h_n, h_u], 1), units=args.emb_dim,
                              activation=None, use_bias=False)
        #
        self.logits = tf.reduce_sum(h_n * target_item, 1)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                   (labels=tf.cast(self.click_label, tf.float32),
                                    logits=self.logits))
        self.opt = tf.train.AdamOptimizer(args.lr).minimize(loss=self.loss, var_list=tf.trainable_variables())

    def train(self):
        print('train')

    def test(self):
        print('test')


model = DGRec()
model.train()
model.test()
