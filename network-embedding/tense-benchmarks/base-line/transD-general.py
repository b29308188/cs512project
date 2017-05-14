#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
from utils import *
from config import *

modelName = 'transD-general'

ll = ctypes.cdll.LoadLibrary   
lib = ll("./init.so")

deg = 2

class TransDModel(object):

	def calc(self, e, t, r):
            return e + tf.reduce_sum(e * t, 1, keep_dims = True) * r

        def compute_regularization(self, entity, sense_embedding, glove_word):
            predict_word = entity
            if self.config.enable_sense:
                predict_word = tf.multiply(entity, sense_embedding)
            difference = predict_word - glove_word
            reg_loss = difference**2
            return tf.reduce_sum(reg_loss)

	def __init__(self, config):

            self.config = config
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
            self.sense_embedding = tf.get_variable(name='sense_embedding', shape = [entity_total, size], initializer = tf.ones_initializer())

            with tf.name_scope("embedding"):
                self.ent_embeddings = tf.get_variable(name = "ent_embedding", shape = [entity_total, size], initializer = config.glove_initializer)
                self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [relation_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
                self.ent_transfer = tf.get_variable(name = "ent_transfer", shape = [entity_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
                self.rel_transfer = tf.get_variable(name = "rel_transfer", shape = [relation_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))

                # the real meaning of the entity
                ent_pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
                ent_pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
                ent_pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
                # the vector for projection
                pos_h_t = tf.nn.embedding_lookup(self.ent_transfer, self.pos_h)
                pos_t_t = tf.nn.embedding_lookup(self.ent_transfer, self.pos_t)
                pos_r_t = tf.nn.embedding_lookup(self.rel_transfer, self.pos_r)

                # the real meaning of the entity
                ent_neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
                ent_neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
                ent_neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
                # the vector for projection
                neg_h_t = tf.nn.embedding_lookup(self.ent_transfer, self.neg_h)
                neg_t_t = tf.nn.embedding_lookup(self.ent_transfer, self.neg_t)
                neg_r_t = tf.nn.embedding_lookup(self.rel_transfer, self.neg_r)

                pos_h_e = self.calc(ent_pos_h_e, pos_h_t, pos_r_t)
                pos_t_e = self.calc(ent_pos_t_e, pos_t_t, pos_r_t)
                neg_h_e = self.calc(ent_neg_h_e, neg_h_t, neg_r_t)
                neg_t_e = self.calc(ent_neg_t_e, neg_t_t, neg_r_t)

            with tf.name_scope('regularization'):

                pos_sense_h_e = tf.nn.embedding_lookup(self.sense_embedding, self.pos_h) 
                pos_sense_t_e = tf.nn.embedding_lookup(self.sense_embedding, self.pos_t) 

                neg_sense_h_e = tf.nn.embedding_lookup(self.sense_embedding, self.neg_h) 
                neg_sense_t_e = tf.nn.embedding_lookup(self.sense_embedding, self.neg_t) 

                reg_pos_glove_h_e = tf.nn.embedding_lookup(self.glove_data, self.pos_h)
                reg_pos_glove_t_e = tf.nn.embedding_lookup(self.glove_data, self.pos_t) 

                reg_neg_glove_h_e = tf.nn.embedding_lookup(self.glove_data, self.neg_h)
                reg_neg_glove_t_e = tf.nn.embedding_lookup(self.glove_data, self.neg_t)

            if config.L1_flag:
                pos = tf.reduce_sum(abs(pos_h_e + ent_pos_r_e - pos_t_e), 1, keep_dims = True)
                neg = tf.reduce_sum(abs(neg_h_e + ent_neg_r_e - neg_t_e), 1, keep_dims = True)
            else:
                pos = tf.reduce_sum((pos_h_e + ent_pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
                neg = tf.reduce_sum((neg_h_e + ent_neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)			

            reg_loss_pos_h = self.compute_regularization(ent_pos_h_e, pos_sense_h_e, reg_pos_glove_h_e)
            reg_loss_pos_t = self.compute_regularization(ent_pos_t_e, pos_sense_t_e, reg_pos_glove_t_e)
            reg_loss_neg_h = self.compute_regularization(ent_neg_h_e, neg_sense_h_e, reg_neg_glove_h_e)
            reg_loss_neg_t = self.compute_regularization(ent_neg_t_e, neg_sense_t_e, reg_neg_glove_t_e)

            reg_loss = reg_loss_pos_h + reg_loss_pos_t + reg_loss_neg_h + reg_loss_neg_t

            with tf.name_scope("output"):
                self.lossR = config.reg_rate*(reg_loss)
                self.lossL = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))
                self.loss = self.lossL+self.lossR

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
                trainModel = TransDModel(config = config)

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

            initRes, initL, initR = None, None, None
            for times in range(config.trainTimes):
                res, lossL_t = 0.0, 0.0
                lossR_t = 0.0
                for batch in range(config.nbatches):
                    lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config.batch_size)
                    loss, lossL, lossR = train_step(ph, pt, pr, nh, nt, nr)
                    res, lossL_t, lossR_t = res + loss, lossL_t + lossL, lossR_t + lossR

                    current_step = tf.train.global_step(sess, global_step)
                if initRes is None:
                    initRes, initL = res, lossL_t
                    initR = lossR_t
                config.printLoss(times, res, initRes, lossL, lossR)

                snapshot = trainModel.ent_embeddings.eval()
                if times%50 == 0:
                    config.writeEmbedding(snapshot)

if __name__ == "__main__":
    tf.app.run()

