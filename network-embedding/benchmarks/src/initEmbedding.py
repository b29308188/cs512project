"""
The program generates a file of init embeddings for 
entity and rlations. 
User can provide prepared emedding files. 

"""
import os, sys
import numpy as np
import argparse


class Embedder(object):
    def __init__(self, inputDir, outputDir, entityFile_n, relationFile_n, eDimension, rDimension):
        """
        Assume that inputDir contains entity2id.txt, relation2id.txt, network.txt
        """
        self.inputDir = inputDir
        self.outputDir = outputDir
        if not os.path.exists(outputDir):
            os.mkdir(self.outputDir)

        self.entityFile_n = entityFile_n
        self.relationFile_n = relationFile_n

        self.eDimension = 100
        if eDimension is not None:
            self.eDimension = eDimension
        self.rDimension = self.eDimension
        if rDimension is not None:
            self.rDimension = rDimension
        
        # id mapping
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}

        # vectors
        self.relation_vec = []
        self.entity_vec = []

    def readIDData2ID(self, fileName, data2id, id2data):
        """
        Read id mapping for either relation or entity embedding.
        It will store data->ID and ID->data in two separate dictionaries.
        It will return the number of entries read in the file.
        """
        dataFile =  os.path.join(self.inputDir, fileName)
        numEntries = 0
        with open(dataFile, 'r') as fileHandler:
            for line in fileHandler:
                arr = line.split()
                data, dataID = arr[0], arr[1]
                data2id[data] = dataID
                id2data[dataID] = data
                numEntries += 1

        return numEntries

    def checkExistence(self, dataDict, searchKey):
        if not searchKey in dataDict:
            print("missing entity: {} ".format(searchKey), file=sys.stderr)
            dataDict[searchKey] = self.numEntities
            
    def readData(self):
        self.readIDData()
        self.readNetworkData("network.data")
        

    def readNetworkData(self, fileName): 
        """
        Read the network data (h,t,r). Where h and t are entities and 
        r is the relation between h and t.
        """
        dataFile = os.path.join(self.inputDir, fileName)
        with open(dataFile, 'r')  as fileHandler:
            for line in fileHandler:
                arr = line.split()
                head, tail, relation = arr[0], arr[1], arr[2]
                if not head in self.entity2id:
                    dataDict[head] = self.numEntities
                    self.numEntities += 1
                if not tail in self.entity2id:
                    dataDict[tail] = self.numEntities
                    self.numEntities += 1
                self.checkExistence(self.entity2id, head)
                self.checkExistence(self.entity2id, tail)

                if relation not in self.relation2id:
                    self.relation2id[relation] = self.numRelations
                    self.numRelations += 1
    
    def readIDData(self):
        """
        Read ID mapping
        """
        self.numRelations = self.readIDData2ID('relation2id.txt', self.relation2id, self.id2relation)
        self.numEntities = self.readIDData2ID('entity2id.txt', self.entity2id, self.id2entity)
    
    def getEmbeddingRange(self, dimension):
        """
        Return (r_min, r_max) that is used for random embedding generation
        """
        return -6/np.sqrt(dimension), 6/np.sqrt(dimension)
        
    def generateEmbedding(self):
        """
        Generate Embedding if the user does not supply embedding files
        """
        if self.relationFile_n is None:
            rMin, rMax = self.getEmbeddingRange(self.rDimension)
            sizeTuple = (self.numRelations, self.rDimension)
            self.relationEmbeddings = np.random.uniform(rMin, rMax,sizeTuple)

        if self.entityFile_n is None:
            eMin, eMax = self.getEmbeddingRange(self.eDimension)
            sizeTuple = (self.numEntities, self.eDimension)
            self.entityEmbeddings = np.random.uniform(eMin, eMax, sizeTuple)

    def writeEmbedding(self):
        """
        Write the relation embeddings to outputdir/relationEmbedding.data
        Write the entity embeddings to outputdir/entityEmbedding.data
        """
        relationFile = os.path.join(self.outputDir, 'relationEmbedding.data')
        entityFile = os.path.join(self.outputDir, 'entityEmbedding.data')
        print('num relations: {} num entities: {}'.format( self.numRelations, self.numEntities))

        with open(relationFile, 'w') as relationHandler:
            for eachVec in self.relationEmbeddings:
                nextStr = ''
                for element in eachVec:
                    nextStr += ' {}'.format(element)
                print(nextStr, file=relationHandler)

        with open(entityFile, 'w') as entityHandler:
            for eachVec in self.entityEmbeddings:
                nextStr = ''
                for element in eachVec:
                    nextStr += ' {}'.format(element)
                print(nextStr, file=entityHandler)

def getArgparser():
    parser = argparse.ArgumentParser(description = 'Generate initial embedding based on given embedding or random generate it.')
    parser.add_argument('dataDir',  help='directory of network data and the id mappings')
    parser.add_argument('outputDir', help='directory for output init embedding') 
    parser.add_argument('-e', '--entityFile', metavar='entityFile', type=str, help='path to a file of entity embedding. If the flag is on, entity embedding will be generated based on this file.')
    parser.add_argument('-r', '--relationFile', metavar='relationFile', type=str, help='path to a file of relation embedding') 
    parser.add_argument('-d', '--dimension', metavar='entityDimension', type=int, help='specify the dimension of embedding. This will also be the value of relationDimension if the user does not specify --relationDimension. If the user does not specify dimension, default dimension is 100.')
    parser.add_argument('-rd', '--relationDimension', metavar='relationDimension', type=int, help='the dimensionality of relation embeddings')
    return parser

if __name__ == '__main__':
    parser=getArgparser()
    args = parser.parse_args()
    
    embedder = Embedder(args.dataDir, args.outputDir, args.entityFile, args.relationFile, args.dimension, args.relationDimension)
    embedder.readData()
    embedder.generateEmbedding()
    embedder.writeEmbedding()

