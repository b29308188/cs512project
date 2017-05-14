from __future__ import print_function
import os, sys
import shutil

def countLine(inputFile):
    counter = 0
    with open(inputFile, 'r') as fileHandler:
        for line in fileHandler:
            counter += 1
    return counter

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def checkExistenceExit(fileName):
    if not os.path.exists(fileName):
        eprint('path: {} doesnt exist'.format(fileName))
        exit(2)

def readTriplesFromFile(fileName):
    checkExistenceExit(fileName)
    tripleList = []
    with open(fileName, 'r') as fileHandler:
        for line in fileHandler:
            arr = line.strip().split()
            head, tail, relation = arr[0], arr[1], arr[2]

            tripleList.append([head, tail, relation])
    return tripleList

def readListFromFile(fileName):
    checkExistenceExit(fileName)
    fileList = []
    with open(fileName, 'r') as fileHandler:
        for line in fileHandler:
            arr = line.strip().split()
            key = arr[0]
            val = arr[1]
            if len(arr) > 2:
                val = arr[1:]
            fileList.append([key, val])
    return fileList

def readDictFromFile(fileName):
    """
    Assume 
    Key Val
    """
    checkExistenceExit(fileName)

    fileDict = {}
    with open(fileName, 'r') as fileHandler:
        for line in fileHandler:
            arr = line.strip().split()
            key = arr[0]
            val = arr[1]
            if len(arr) > 2:
                val = arr[1:]

            fileDict[key] = val

    return fileDict

def writeListToFile(path, inputList):
    with open(path, 'w') as fileHandler:
        for row in inputList:
            each_line = ''
            for elem in row:
                each_line += '{} '.format(elem)
            each_line = each_line.rstrip()
            print(each_line, file=fileHandler)
def writeDictToFile(inputDict, fileName):

    with open(fileName, 'w') as fileHandler:
        for key, value in inputDict.items():
            print('{} {}'.format(key, value), file=fileHandler)

def removeAndMake(path):
    if  os.path.exists(path):
        print('path {} eixsts, remove it ...'.format(path))
        shutil.rmtree(path)

    os.mkdir(path)

