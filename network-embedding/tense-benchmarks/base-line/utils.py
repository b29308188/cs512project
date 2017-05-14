import numpy as np
import os, sys
import argparse

def writeEmbedding(result, fileName):
    np.savetxt(fileName, result)
    print('finish writing ...')

def checkExistence(path):
    return os.path.exists(path)

def checkExistenceExit( *args):
    for path in args:
        if not checkExistence(path):
            print('path: {} does not exist'.format(path))
            exit(2)

def checkAndCreate(path):
    if not checkExistence(path):
        os.mkdir(path)

def getArgs(modelName):
    parser = argparse.ArgumentParser()
    parser.add_argument('inputDir', help='specify the value of input directory', default='data')
    parser.add_argument('outputDir', help='sepcify the value of output directory', default='{}-results'.format(modelName))
    parser.add_argument('-r', '--reg_deg', help='specify the value of regularization', type=int, default = 10)
    parser.add_argument('-b', '--batches', help='specify the number of batches', type=int, default=100)
    parser.add_argument('--random-init', help='use random initialization instead of glove init',dest='random_init', action='store_true')
    parser.add_argument('--no-random-init', help='use glove initialization', dest='random_init', action = 'store_false')
    parser.set_defaults(random_init = False)
    args = parser.parse_args()
    return args

