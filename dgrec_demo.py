#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import roc_auc_score
import pickle, argparse, datetime, os, sys, itertools, time, scipy, math

np.random.seed(0)
tf.set_random_seed(2)
#
parser = argparse.ArgumentParser()
parser.add_argument('-MAX_TIME_STEP', type=int, default=10)
parser.add_argument('-batch_size', type=int, default=12)
parser.add_argument('-user_num', type=int, default=1000)
parser.add_argument('-item_num', type=int, default=1000)
parser.add_argument('-emb_dim', type=int, default=128)
parser.add_argument('-max_fri_num', type=int, default=10)
parser.add_argument('-layer_num', type=int, default=10)
parser.add_argument('-lr', type=int, default=1e-3)
args = parser.parse_args()


class DGRec():
    def __init__(self):
        stdv = 1.0 / math.sqrt(args.emb_dim)
        self.initializer = tf.random_uniform_initializer(-stdv, stdv)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()

    def build_graph(self):
        # aria
        # self.target_user = tf.placeholder(tf.int32, [args.batch_size], 'target_user')
        self.sess_seq = tf.placeholder(tf.int32, [args.batch_size, None], 'sess_seq')
        self.sess_len = tf.placeholder(tf.int32, [args.batch_size], 'sess_len')
        self.target_item = tf.placeholder(tf.int32, [args.batch_size], 'target_item')
        self.click_label = tf.placeholder(tf.int32, [args.batch_size], 'click_label')
        self.friends = tf.placeholder(tf.int32, [args.batch_size, args.max_fri_num], 'friends')
        self.friends_sess_seq = tf.placeholder(tf.int32, [args.batch_size, args.max_fri_num, args.MAX_TIME_STEP],
                                               'friends_sess_seq')
        self.friends_sess_len = tf.placeholder(tf.int32, [args.batch_size, args.max_fri_num],
                                               'friends_sess_len')
        self.friends_num = tf.placeholder(tf.float32, [args.batch_size], 'friends_num')
        # lookup dict
        self.user_embed_dict = tf.get_variable('user_embed_dict', [args.user_num, args.emb_dim],
                                               tf.float32, self.initializer)
        self.item_embed_dict = tf.get_variable('item_embed_dict', [args.item_num, args.emb_dim],
                                               tf.float32, self.initializer)
        #
        sess_seq = tf.nn.embedding_lookup(self.item_embed_dict, self.sess_seq)
        target_item = tf.nn.embedding_lookup(self.item_embed_dict, self.target_item)
        # lstm
        cell = tf.nn.rnn_cell.LSTMCell(args.emb_dim, activation=lambda x: tf.tanh(x), name='lstm')
        zero_state = cell.zero_state(args.batch_size, tf.float32)
        # h_n - 玩家的short term interest - 公式(1)
        _, state = tf.nn.dynamic_rnn(cell, sess_seq, sequence_length=self.sess_len, initial_state=zero_state)
        h_n = state[1]
        # s_k_s - 好友short term interest - 公式(3)
        all_friends_seq = tf.reshape(self.friends_sess_seq, [-1, args.MAX_TIME_STEP])
        friends_sess_seq = tf.nn.embedding_lookup(self.item_embed_dict, all_friends_seq)
        zero_state_f = cell.zero_state(args.batch_size * args.max_fri_num, tf.float32)
        _, state_f = tf.nn.dynamic_rnn(cell, friends_sess_seq,
                                       sequence_length=tf.reshape(self.friends_sess_len, [-1]),
                                       initial_state=zero_state_f)
        s_k_s = state_f[1]
        # s_k_l - 好友long term interest - 公式(4)
        friends = tf.nn.embedding_lookup(self.user_embed_dict, tf.reshape(self.friends, [-1]))
        # s_k - 好友final interest - 公式(5)
        s_k = tf.layers.dense(inputs=tf.concat([s_k_s, friends], 1), units=args.emb_dim,
                              activation=tf.nn.relu, use_bias=False)
        s_k = tf.reshape(s_k, [args.batch_size, args.max_fri_num, -1])
        # dynamic feature graph
        h_u = h_n
        for l_idx in range(args.layer_num):
            # similarity - 公式(6)
            a_uk = tf.matmul(s_k, tf.reshape(h_u, [args.batch_size, args.emb_dim, 1]))
            # padding的好友权重置为0
            friends_mask = tf.cast(tf.sequence_mask(self.friends_num, args.max_fri_num), tf.float32)
            a_uk = tf.nn.softmax(tf.squeeze(a_uk) * friends_mask, 1)
            # combine friends interest - 公式(7)
            h_u_ = tf.reduce_sum(tf.expand_dims(a_uk, 2) * s_k, 1)
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
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        print('train')

    def test(self):
        print('test')


model = DGRec()
model.train()
model.test()
