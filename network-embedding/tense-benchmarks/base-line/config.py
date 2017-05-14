import os, sys
import numpy as np
from utils import *
import argparse
import tensorflow as tf
import csv


class Config(object):
    def __init__(self, inputDir, outputDir):
        print('input directory is: {}\noutput directory is: {}'.format(inputDir, outputDir))
        self.inputDir, self.outputDir = inputDir, outputDir
        checkExistenceExit(self.inputDir)
        checkAndCreate(self.outputDir)

        self.relationFile = os.path.join(inputDir, 'relation2id.txt')
        self.entityFile = os.path.join(inputDir, 'entity2id.txt')
        self.tripleFile = os.path.join(inputDir, 'triple2id.txt')

        self.initEmbeddingFile = os.path.join(inputDir, 'initSenseEmbedding.txt')
        self.embeddingFile = None

        checkExistenceExit(self.relationFile, self.entityFile, self.tripleFile)


        self.random_init = False
        self.L1_flag = True
        self.dimension_e = 300
        self.dimension_r = 100
        self.nbatches = 100
        self.batch_size = 100
        self.entity, self.relation = 0, 0 
        self.trainTimes = 3000
        self.margin = 1.0

        self.reg_deg = 10
        self.reg_rate = 10**(-self.reg_deg)

        self.senseFile = None
        self.enable_sense = False

        self.learning_curve_file = None

    def set_regularization(self, deg):
        self.reg_deg = deg
        self.reg_rate = 10**(-self.reg_deg)
        self.embeddingFile = os.path.join(self.outputDir, 'reg-{}.txt'.format(self.reg_deg))
        self.senseFile = os.path.join(self.outputDir, 'reg-{}-sense.txt'.format(self.reg_deg))

    def readEmbeddings(self):
        if self.random_init:
            self.glove_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            return 

        checkExistenceExit(self.initEmbeddingFile)
        self.initEmbeddings = None
        with open(self.initEmbeddingFile, 'r') as fileHandler:
            firstLine = fileHandler.readline()
            self.initEmbeddings = np.zeros((self.entity,self.dimension_e))
            counter = 0
            for line in fileHandler:
                arr = line.strip().split()
                idx = int(arr[0])
                arr_list = list(map(float, arr[1:]))
                self.initEmbeddings[idx] = arr_list
                counter += 1

        self.glove_initializer = tf.constant_initializer(self.initEmbeddings)

    def openCsvFile(self, fileName):
        fieldnames = ['step', 'loss']
        self.learning_curve_file = open(fileName, 'w')
        self.learning_writer = csv.DictWriter(self.learning_curve_file, fieldnames=fieldnames)

    def printLoss(self, times, lossT, initlossT, lossL, lossR):
        print('epoch: {} loss: {} percentage:{:.3f}%'.format(times, lossT, lossT/initlossT*100))
        print('lossL: {} ({:.3f}%) lossR: {} ({:.3f}%)'.format(lossL, lossL/lossT, lossR, lossR/lossT))
        if self.learning_curve_file is None:
            fileName = os.path.join(self.outputDir, 'curve-reg-{}.csv'.format(self.reg_deg))
            self.openCsvFile(fileName)
            self.learning_writer.writeheader()
        self.learning_writer.writerow({'step': str(times), 'loss':str(lossT)})

    def writeEmbedding(self, snapshot):
        writeEmbedding(snapshot, self.embeddingFile)
        if not self.learning_curve_file is None:
            self.learning_curve_file.flush()


    def Print(self):
        print('inputs: {} {} {}'.format(self.relationFile, self.entityFile, self.tripleFile))
        print('init embedding: {}'.format(self.initEmbeddingFile))
        print('output: {}'.format(self.embeddingFile))
        print('num entities: {} dimensions: {}'.format(self.entity, self.dimension_e))
        print('num relations: {} dimensions: {}'.format(self.relation, self.dimension_r))
        print('num batches: {}'.format(self.nbatches))
        print('regularization deg: {} rate: {}'.format(self.reg_deg, self.reg_rate))
        print('random initialization: {}'.format(self.random_init))

