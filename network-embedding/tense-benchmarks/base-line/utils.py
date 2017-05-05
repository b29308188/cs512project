import numpy as np
import os, sys

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
