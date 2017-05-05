#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
from utils import writeEmbedding

ll = ctypes.cdll.LoadLibrary   
lib = ll("./init.so")

deg = 10
outputDir = 'transR-results' 
embeddingFileName = '{}/reg-{}'.format(outputDir, deg)
modelFileName = '{}/model.vec-adam-{}'.format(outputDir, deg)

class Config(object):
    def __init__(self):
        self.L1_flag = True
        self.hidden_sizeE = 300
        self.hidden_sizeR = 100
        self.nbatches = 100
        self.entity = 0
        self.relation = 0
        self.trainTimes = 3000
        self.margin = 1.0
        self.reg_rate = 1000
        self.learning_rate = 0.001
        self.glove_initializer = None

    def readEmbeddings(self):
        fileName = './data/initSenseEmbedding.txt'
        if not os.path.exists(fileName):
            print('{} does not exist'.format(name))
            exit(2)

        self.initEmbeddings = None
        with open(fileName, 'r') as fileHandler:
            firstLine = fileHandler.readline()
            self.initEmbeddings = np.zeros((self.entity,self.hidden_size))
            counter = 0
            for line in fileHandler:
                arr = line.strip().split()
                idx = int(arr[0])
                arr_list = list(map(float, arr[1:]))
                self.initEmbeddings[idx] = arr_list
                counter += 1

        self.glove_initializer = tf.constant_initializer(self.initEmbeddings)

class TransRModel(object):

    def __init__(self, config):

        entity_total = config.entity
        relation_total = config.relation
        batch_size = config.batch_size
        sizeE = config.hidden_sizeE
        sizeR = config.hidden_sizeR
        margin = config.margin

        self.glove_data = tf.get_variable(name='glove_embedding', shape = [entity_total, sizeE], trainable=False, initializer = config.glove_initializer)

        with tf.name_scope("read_inputs"):
            self.pos_h = tf.placeholder(tf.int32, [batch_size])
            self.pos_t = tf.placeholder(tf.int32, [batch_size])
            self.pos_r = tf.placeholder(tf.int32, [batch_size])

            self.neg_h = tf.placeholder(tf.int32, [batch_size])
            self.neg_t = tf.placeholder(tf.int32, [batch_size])
            self.neg_r = tf.placeholder(tf.int32, [batch_size])

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name = "ent_embedding", shape = [entity_total, sizeE], initializer = config.glove_initializer)
            self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [relation_total, sizeR], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.rel_matrix = tf.get_variable(name = "rel_matrix", shape = [relation_total, sizeE * sizeR], initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        with tf.name_scope('lookup_embeddings'):
            pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h), [-1, sizeE, 1])
            pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t), [-1, sizeE, 1])
            pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r), [-1, sizeR, 1])

            neg_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h), [-1, sizeE, 1])
            neg_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t), [-1, sizeE, 1])
            # it was of [-1, sizeE, 1]. Dont know why they take embedding in the shpae of entity
            neg_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r), [-1, sizeR, 1])			
            
            matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, self.neg_r), [-1, sizeR, sizeE])

            pos_h_e = tf.reshape(tf.batch_matmul(matrix, pos_h_e), [-1, sizeR])
            pos_t_e = tf.reshape(tf.batch_matmul(matrix, pos_t_e), [-1, sizeR])
            pos_r_e = tf.reshape(pos_r_e, [-1, sizeR])
   #         pos_r_e = tf.reshape(tf.batch_matmul(matrix, pos_r_e), [-1, sizeR])
            neg_h_e = tf.reshape(tf.batch_matmul(matrix, neg_h_e), [-1, sizeR])
            neg_t_e = tf.reshape(tf.batch_matmul(matrix, neg_t_e), [-1, sizeR])
 #           neg_r_e = tf.reshape(tf.batch_matmul(matrix, neg_r_e), [-1, sizeR])
            neg_r_e = tf.reshape(neg_r_e, [-1, sizeR])
        
        with tf.name_scope('regularization'):
            reg_pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
            reg_pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
            reg_pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)

            reg_neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
            reg_neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
            reg_neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

            reg_pos_glove_h_e = tf.nn.embedding_lookup(self.glove_data, self.pos_h)
            reg_pos_glove_t_e = tf.nn.embedding_lookup(self.glove_data, self.pos_t) 

            reg_neg_glove_h_e = tf.nn.embedding_lookup(self.glove_data, self.neg_h)
            reg_neg_glove_t_e = tf.nn.embedding_lookup(self.glove_data, self.neg_t)

        if config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims = True)
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)			
        
        reg_loss_pos_h = (reg_pos_glove_h_e - reg_pos_h_e)*(reg_pos_glove_h_e - reg_pos_h_e)
        reg_loss_pos_t = (reg_pos_glove_t_e - reg_pos_t_e)*(reg_pos_glove_t_e - reg_pos_t_e)
        reg_loss_neg_h = (reg_neg_glove_h_e - reg_neg_h_e)*(reg_neg_glove_h_e - reg_neg_h_e)
        reg_loss_neg_t = (reg_neg_glove_t_e - reg_neg_t_e)*(reg_neg_glove_t_e - reg_neg_t_e)

        reg_loss = tf.reduce_sum(reg_loss_pos_h) + tf.reduce_sum(reg_loss_pos_t) + tf.reduce_sum(reg_loss_neg_h) + tf.reduce_sum(reg_loss_neg_t)

        with tf.name_scope("output"):
            self.lossR = config.reg_rate*(reg_loss)
            self.lossL = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))
            self.loss = self.lossR + self.lossL


def main(args):
    
    inputDir = args[1]
    
    relationFile = os.path.join(inputDir, 'relation2id.txt')
    entityFile = os.path.join(inputDir, 'entity2id.txt')
    tripleFile = os.path.join(inputDir, 'triple2id.txt')

    checkExistenceExit(relationFile, entityFile, tripleFile)

    lib.init(relationFile, entityFile, tripleFile)
    config = Config(inputDir)
    config.relation = lib.getRelationTotal()
    config.entity = lib.getEntityTotal()
    config.batch_size = lib.getTripleTotal() / config.nbatches
    config.readEmbeddings()

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer(uniform = False)
            with tf.variable_scope("model", reuse=None, initializer = initializer):
                trainModel = TransRModel(config = config)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.001)
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
                res, lossL_t, lossR_t = 0.0, 0.0, 0.0
                for batch in range(config.nbatches):
                    lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config.batch_size)
                    loss, lossL, lossR = train_step(ph, pt, pr, nh, nt, nr)
                    res, lossL_t, lossR_t = res + loss, lossL_t + lossL, lossR_t + lossR

                    current_step = tf.train.global_step(sess, global_step)
                if initRes is None:
                    initRes, initL, initR = res, lossL_t, lossR_t 

                print('epoch: {} loss:{:.3f} percentage:{:.3f}'.format(times, res, res/initRes))
                print('lossL: {} ({:.3f}%) lossR: {} ({:.3f})%'.format(lossL_t, lossL_t/res*100, lossR_t, lossR_t/res*100)) 

                snapshot = trainModel.ent_embeddings.eval()
                if times%50 == 0:
                    writeEmbedding(snapshot, embeddingFileName)
                    saver.save(sess, modelFileName)

if __name__ == "__main__":
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    tf.app.run()

