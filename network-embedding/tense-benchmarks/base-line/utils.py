import numpy as np

def writeEmbedding(result, fileName):
    np.savetxt(fileName, result)
    print('finish writing ...')
