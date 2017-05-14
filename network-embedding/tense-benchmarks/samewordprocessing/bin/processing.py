import os, sys
import numpy


def getWord(entity):
    word = entity.split('.')[0]
    return word

def getSameWordDict(entityFile):
    revDict = {}
    with open(entityFile, 'r') as fileHandler:
        for line in fileHandler:
            entity, idx = line.strip().split()
            word = getWord(entity)
            if word in revDict:
                revDict[word] += '{} '.format(idx)
            else:
                revDict[word] = '{} '.format(idx)
    return revDict

def getIndexMatrix(entityFile, sameWordDict):

    revMat = []
    with open(entityFile, 'r') as fileHandler:
        for line in fileHandler:
            entity, idx = line.strip().split()
            word = getWord(entity)
            relatedIDs = sameWordDict[word]
            revMat.append(relatedIDs)
    return revMat
            
def writeIndexMat(outputFile, indexMat):

    with open(outputFile, 'w') as fileHandler:
        for row in indexMat:
            print(row, file=fileHandler)


def main(inputDir,outputDir):
    entityFile = os.path.join(inputDir, 'entity2id.txt')
    sameWordIdxDict = getSameWordDict(entityFile)
    indexMat = getIndexMatrix(entityFile, sameWordIdxDict)

    outputFile = os.path.join(outputDir, 'sameword2id.txt')
    writeIndexMat(outputFile, indexMat)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: python processing.py rawDataDir outputDir', file=sys.stderr)
        exit(1)
    if not os.path.exists(sys.argv[1]):
        print('{} doesnt exist'.format(sys.argv[1]), file=sys.stderr)
        exit(1)
    if not os.path.exists(sys.argv[2]):
        print('{} does not exist'.format(sys.argv[2]), file=sys.stderr)
        exit(1)

    inputDir, outputDir = sys.argv[1:]
    main(inputDir, outputDir)
