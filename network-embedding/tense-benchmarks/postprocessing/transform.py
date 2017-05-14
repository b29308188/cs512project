from __future__ import print_function
import os, sys
import numpy as np
import progressbar


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
    print('writing embedding out ...' )
    num_entity = len(embeddings)
    with open(outputFilePath, 'w') as fileHandler:
        print('{} {}'.format(num_entities, dimension), file=fileHandler)
        counter = 0
        bar = progressbar.ProgressBar(maxval=num_entity,
                    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        for row in embeddings:
            entityName = mapping[counter]
            line = row2str(row)
            counter += 1
            print('{} {}'.format(entityName, line), file=fileHandler)
            bar.update(counter)
        bar.finish()
    print('done writing files ...')


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('usage: python3 transform.py [embedding file] [entity id mapping] [output file path]')
        exit(1)

    embeddingFile, idMappingFile, outputFilePath = sys.argv[1:]

    embeddings = readMatrix(embeddingFile)
    idMapping = readMapping(idMappingFile)
    writeEmbeddings(embeddings, idMapping, outputFilePath)
