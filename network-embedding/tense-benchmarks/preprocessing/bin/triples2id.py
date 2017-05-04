import os, sys
from utils import *


def getMappingDictionary(fileName):
    myDict = {}
    with open(fileName, 'r') as fileHandler:
        for line in fileHandler:
            mapping = line.strip().split()
            value, idx = mapping[0], mapping[1]
            myDict[value] = idx

    return myDict

def transformTriples(tripleFile, e2id, r2id):
    triples = []
    with open(tripleFile, 'r') as fileHandler:
        for line in fileHandler:
            head, tail, rel = line.strip().split()
            headID, tailID, relID = e2id[head], e2id[tail], r2id[rel]
            triples.append([headID, tailID, relID])

    return triples

def writeRelations(outputFile, relations):
    numRels = len(relations)
    with open(outputFile, 'w') as fileHandler:
        print(numRels, file=fileHandler)
        for rel in relations:
            print('{} {} {}'.format(rel[0], rel[1], rel[2]), file=fileHandler)

if __name__ == '__main__':

    if len(sys.argv) != 5:
        eprint('usage: python triples2id.py entityFile relationFile tripleRawFile outputFile')
        exit(2)

    entityDictFile, relationDictFile, triplesRawFile, outputFile  = sys.argv[1:]

    checkExistenceExit(entityDictFile)
    checkExistenceExit(relationDictFile)
    checkExistenceExit(triplesRawFile)

    entity2id = getMappingDictionary(entityDictFile)
    relation2id = getMappingDictionary(relationDictFile)
    
    relations = transformTriples(triplesRawFile, entity2id, relation2id)

    writeRelations(outputFile, relations)
