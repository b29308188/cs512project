import os, sys
import numpy as np


def readMatrix(embeddingFile):
    print('reading {} ...'.format(embeddingFile))
    embeddings = np.loadtxt(embeddingFile)
    return embeddings

def readMapping(idMappingFile):
    print('reading id mapping from {} ...'.format(idMappingFile))
    revDict = {}
    with open(idMappingFile) as fileHandler:
        numLines = fileHandler.readline()
        numLines = int(numLines)

        for line in fileHandler:
            arr = line.strip().split()
            entityName, entityID = arr[0], int(arr[1])
            revDict[entityID] = entityName
    return revDict

def row2str(row):
    line = ''
    for elem in row:
        line += '{} '.format(str(elem))
    line = line.rstrip()
    return line


def writeEmbeddings(embeddings, mapping, outputFilePath):
    num_entities, dimension = embeddings.shape

    with open(outputFilePath, 'w') as fileHandler:
        print('{} {}'.format(num_entities, dimension), file=fileHandler)
        counter = 0
        for row in embeddings:
            entityName = mapping[counter]
            line = row2str(row)
            counter += 1
            print('{} {}'.format(entityName, line), file=fileHandler)
    print('done writing files ...')


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('usage: python3 transform.py [embedding file] [output file path]')
        exit(1)

    embeddingFile, outputFilePath = sys.argv[1:]
    idMappingFile = './data/entity2id.txt'

    embeddings = readMatrix(embeddingFile)
    idMapping = readMapping(idMappingFile)
    writeEmbeddings(embeddings, idMapping, outputFilePath)
