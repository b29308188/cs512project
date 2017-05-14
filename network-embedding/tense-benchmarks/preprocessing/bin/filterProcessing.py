from __future__ import print_function
from progressbar import *
import os, sys
from utils import *
from shutil import rmtree

def addPrefixAndPrint(inputFile, outputFile):
    of = open(outputFile, 'w')

    print('counting lines in inputFile {}'.format(inputFile))
    numLines = countLine(inputFile)

    print('writing to outputfile: {} ...'.format(outputFile))
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=numLines).start()
    with open(inputFile, 'r') as fileHandler:
        counter = 0
        print(numLines, file=of)
        for line in fileHandler:
            line = line.rstrip()
            counter += 1
            print(line, file=of)
            pbar.update(counter)
    pbar.finish()

def processEntity(entityFile, gloveDict, entityOutput):
    
    print('counting lines in inputFile: {}'.format(entityFile))
    numLines = countLine(entityFile)
    print('count those missing entities ... '.format(entityFile))
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=numLines).start()
    inputDict = readDictFromFile(entityFile)
    counter, total = 0, 0
    of = open(entityOutput, 'w')
    newEntityDict = {}
    for name, idx in inputDict.items():
        if name in gloveDict:
            nextLine = '{} {}'.format(name, counter)
            print(nextLine, file=of)
            newEntityDict[name] = counter
            counter += 1
        total += 1
        pbar.update(total)
    pbar.finish()
    return newEntityDict

def checkKeyInDict(key, myDict):
    if not key in myDict:
        print('{} not in the dictionary'.format(key))
        exit(2)

def  processNetwork(networkFile, gloveDict, entityDict, relationDict, networkOutput):
    print('counting lines in inputFile: {}'.format(entityFile))
    numLines = countLine(networkFile)

    print('count those missing entities ... '.format(entityFile))
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=numLines).start()
    outputList = []
    counter = 0
    of = open(networkOutput, 'w')
    missing = 0 
    with open(networkFile, 'r') as fileHandler:
        for line in fileHandler:
            line = line.rstrip()
            counter += 1
            head, tail, relation = line.split() 
            if not relation in relationDict:
                print('relation: {} is not in the dictionary.'.format(relation), file=sys.stderr)
                exit(2)
            if head in gloveDict and tail in gloveDict:
                checkKeyInDict(head, entityDict)
                checkKeyInDict(tail, entityDict)

                toWrite = '{} {} {}'.format(entityDict[head], entityDict[tail], 
                            relationDict[relation])
                print(toWrite, file = of)
            else:
                missing += 1
            pbar.update(counter)
    pbar.finish()
    print('total: {} missing: {}'.format(counter, missing))

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: python filterProcessing.py inputDir gloveDictionary outputDir')
        exit(2)
    inputDir, gloveFile, outputDir = sys.argv[1:]
    checkExistenceExit(inputDir)
    checkExistenceExit(gloveFile) 

    if os.path.exists(outputDir):
        print('outputDir {} exists, remove it ...'.format(outputDir))
        rmtree(outputDir)
        os.mkdir(outputDir)

    entityFile = os.path.join(inputDir, 'entity2id.txt')
    relationFile = os.path.join(inputDir, 'relation2id.txt')
    networkFile = os.path.join(inputDir, 'triples.txt')

    checkExistenceExit(entityFile)
    checkExistenceExit(relationFile) 
    checkExistenceExit(networkFile)

    gloveDict = readDictFromFile(gloveFile)

    entityOutput = os.path.join(outputDir, 'entity2id.txt')
    entityOutputTmp = os.path.join(outputDir, 'entityTmp')
    relationOutput = os.path.join(outputDir, 'relation2id.txt')
    networkOutput = os.path.join(outputDir, 'triple2id.txt')
    networkTmp = os.path.join(outputDir, 'networkTmp')

    # because there are missing entities, we have to reset entity ids
    newEntityDict = processEntity(entityFile, gloveDict, entityOutputTmp)
    addPrefixAndPrint(entityOutputTmp, entityOutput)
    os.remove(entityOutputTmp)
    addPrefixAndPrint(relationFile, relationOutput)

    entityDict = readDictFromFile(entityFile)
    relationDict = readDictFromFile(relationFile)
    # notice that we should use new entity ID mapping
    processNetwork(networkFile, gloveDict, newEntityDict, relationDict, networkTmp)
    addPrefixAndPrint(networkTmp, networkOutput)
    os.remove(networkTmp)
