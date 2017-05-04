from __future__ import print_function
import os, sys
from utils import *
import collections
import numpy as np 
from progressbar import *

def generateRandomEmbeddings(dimension):
    eMin, eMax = -6/np.sqrt(dimension), 6/np.sqrt(dimension)
    sizeTuple = (1, dimension)
    embedding = np.random.uniform(eMin, eMax, sizeTuple)
    return embedding[0]

def createMapping(inputDict, gloveDict):
    
    embeddingDict = {}
    counter = 0 
    for name, idx in inputDict.items():
        embeddings = None
        if name in gloveDict:
            embeddings = gloveDict[name]
            embeddingDict[int(idx)] = embeddings
        else:
            counter += 1
            embeddings = generateRandomEmbeddings(300)
            continue
    if (counter + len(embeddingDict)) != len(inputDict):
        print('incorrect number of missing entities or embedding entities ...')
        exit(2)
    print('total missing: {}'.format(counter))
    
    return embeddingDict

def writeResults(resDict, fileName):
    print('total number of entites to write: {}'.format(len(resDict)))
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(resDict)).start()
    with open(fileName, 'w') as fileHandler:
        print(len(resDict), file=fileHandler)
        counter = 0
        for key, val in resDict.items():
            counter += 1
            line = '{} '.format(key)
            for elem in val:
                line += '{} '.format(elem)
            print(line, file=fileHandler)
            pbar.update(counter)
    pbar.finish()

def readInput(inputFile):
    inputDict = {}
    with open(inputFile, 'r') as fileHandler:
        first_line = fileHandler.readline()
        for line in fileHandler:
            arr = line.strip().split()
            name, idx = arr[0], arr[1]
            inputDict[name] = idx
    return inputDict



if __name__ == '__main__':
    if len(sys.argv) != 4:
        eprint('usage: python3 initEmbedding.py inputFile dictionary outputDir')
        exit(2)

    inputFile, gloveFile, outputDir = sys.argv[1:]
    outputFile = os.path.join(outputDir, 'initSenseEmbedding.txt')

    checkExistenceExit(inputFile)
    checkExistenceExit(gloveFile)
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    print('reading glove data ... ')
    gloveDict = readDictFromFile(gloveFile)
    print('reading input data ... ')
    inputDict = readInput(inputFile)

    print('creating mapping ...')
    results = createMapping(inputDict, gloveDict)

    print('writing results ...') 
    writeResults(results, outputFile)

