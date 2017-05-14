import os, sys
from utils import *
import copy

def extractRelations(network):
    
    triples = readTriplesFromFile(network)
    relationSet = set()
    for triple in triples:
        relation = triple[2]
        relationSet.add(relation)

    counter = 0
    relation2id = []
    for relation in relationSet:
        relation2id.append([relation, counter])
        counter += 1
    return relation2id

def combineRelations(exists, newrelations):

    counter = len(exists)
    combined = copy.deepcopy(exists)
    for relation in newrelations:
        relationName, relationIdx = relation
        combined.append([relationName, relationIdx + counter])
    return combined



if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        print('usage: python addrelations.py [existing relations] [new relations] [output dir]') 
        exit(2)
    
    existingRelations, newNetwork = sys.argv[1], sys.argv[2]
    outputDir = sys.argv[3]
    removeAndMake(outputDir)
    outputFile = os.path.join(outputDir, 'relation2id.txt')

    existingList = readListFromFile(existingRelations)
    newRelations = extractRelations(newNetwork)

    combinedRelations = combineRelations(existingList, newRelations)

    writeListToFile(outputFile, combinedRelations)
    


