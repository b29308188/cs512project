#coding:utf-8
import __future__
import numpy as np
import tensorflow as tf
import os, signal
import time
import datetime
import ctypes
from utils import *
import argparse
from config import *

modelName = 'transE'

ll = ctypes.cdll.LoadLibrary   
lib = ll("./init.so")

snapshot = None
flag = False

def handler(signum, frame):
    print 'cleaning up things and save snapshot... '
    flag = True

class TransEModel(object):

    def __init__(self, config):

        entity_total = config.entity
        relation_total = config.relation
        batch_size = config.batch_size
        size = config.dimension_e
        margin = config.margin

        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])

        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])

        self.glove_data = tf.get_variable(name='glove_embedding', shape = [entity_total, size], trainable=False, initializer = config.glove_initializer)

        with tf.name_scope("embedding"):

            self.ent_embeddings = tf.get_variable(name = "ent_embedding", shape = [entity_total, size],  initializer = config.glove_initializer)
            self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [relation_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))


            pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
            pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)

            neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
            neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

            pos_glove_h_e, pos_glove_t_e,neg_glove_h_e, neg_glove_t_e = None, None, None, None

            pos_glove_h_e = tf.nn.embedding_lookup(self.glove_data, self.pos_h)
            pos_glove_t_e = tf.nn.embedding_lookup(self.glove_data, self.pos_t) 

            neg_glove_h_e = tf.nn.embedding_lookup(self.glove_data, self.neg_h)
            neg_glove_t_e = tf.nn.embedding_lookup(self.glove_data, self.neg_t)

        if config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims = True)
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)            

        self.pos_h_reg_loss = (pos_glove_h_e - pos_h_e)*(pos_glove_h_e -  pos_h_e)
        self.pos_t_reg_loss = (pos_glove_t_e - pos_t_e)*(pos_glove_t_e -  pos_t_e)
        self.neg_h_reg_loss = (neg_glove_h_e - neg_h_e)*(neg_glove_h_e -  neg_h_e)
        self.neg_t_reg_loss = (neg_glove_t_e - neg_t_e)*(neg_glove_t_e -  neg_t_e)

        self.total_reg_loss = tf.reduce_sum(self.pos_h_reg_loss) + tf.reduce_sum(self.pos_t_reg_loss) + tf.reduce_sum(self.neg_h_reg_loss) + tf.reduce_sum(self.neg_t_reg_loss)

        with tf.name_scope("output"):
            self.lossL = tf.reduce_sum(tf.maximum(pos-neg+margin, 0))
            self.lossR = config.reg_rate*(self.total_reg_loss)
            self.loss =  self.lossL + self.lossR


def main(args):
    
    args = getArgs(modelName)
    config = Config(args.inputDir, args.outputDir)
    config.set_regularization(args.reg_deg)
    config.nbatches = args.batches
    config.random_init = args.random_init
    
    lib.init(config.relationFile, config.entityFile, config.tripleFile)

    config.relation = lib.getRelationTotal()
    config.entity = lib.getEntityTotal()
    config.batch_size = lib.getTripleTotal() / config.nbatches
    config.readEmbeddings()
    config.Print()

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer(uniform = False)
            with tf.variable_scope("model", reuse=None, initializer = initializer):
                trainModel = TransEModel(config = config)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer()
            grads_and_vars = optimizer.compute_gradients(trainModel.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            def train_step(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
                feed_dict = {
                    trainModel.pos_h: pos_h_batch,
                    trainModel.pos_t: pos_t_batch,
                    trainModel.pos_r: pos_r_batch,
                    trainModel.neg_h: neg_h_batch,
                    trainModel.neg_t: neg_t_batch,
                    trainModel.neg_r: neg_r_batch
                }
                _, step, loss, lossL, lossR = sess.run(
                    [train_op, global_step, trainModel.loss, trainModel.lossL, trainModel.lossR], feed_dict)
                return loss, lossL, lossR

            ph = np.zeros(config.batch_size, dtype = np.int32)
            pt = np.zeros(config.batch_size, dtype = np.int32)
            pr = np.zeros(config.batch_size, dtype = np.int32)
            nh = np.zeros(config.batch_size, dtype = np.int32)
            nt = np.zeros(config.batch_size, dtype = np.int32)
            nr = np.zeros(config.batch_size, dtype = np.int32)

            ph_addr = ph.__array_interface__['data'][0]
            pt_addr = pt.__array_interface__['data'][0]
            pr_addr = pr.__array_interface__['data'][0]
            nh_addr = nh.__array_interface__['data'][0]
            nt_addr = nt.__array_interface__['data'][0]
            nr_addr = nr.__array_interface__['data'][0]

            initRes = None
            initL, initR = None, None
            for times in range(config.trainTimes):
                res = 0.0
                lossL_t, lossR_t = 0, 0
                for batch in range(config.nbatches):
                    lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config.batch_size)
                    loss, lossL, lossR= train_step(ph, pt, pr, nh, nt, nr)

                    res += loss
                    lossL_t += lossL
                    lossR_t += lossR

                    current_step = tf.train.global_step(sess, global_step)
                    if flag == True:
                        break
                if flag == True:
                    break
                if initRes is None:
                    initRes = res
                config.printLoss(times, res, initRes, lossL, lossR)

                snapshot = trainModel.ent_embeddings.eval()
                if times%50 == 0:
                    config.writeEmbedding(snapshot)


if __name__ == "__main__":
    tf.app.run()

