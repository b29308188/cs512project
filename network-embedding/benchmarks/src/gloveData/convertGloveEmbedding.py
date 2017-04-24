import os, sys
import numpy as np
import random

eidF = open('eid.small', 'r')
networkF = open('small.txt', 'r')

gloveDict = {}

with open('./glove.6B.300d.txt', 'r') as gloveHandler:
    for line in gloveHandler:
        arr =line.strip().split() 
        word = arr[0]
        embedding = arr[1:]
        gloveDict[word] = embedding

with open('enittyEmbedding.data', 'w') as outputHandler:
    for line in eidF:
        arr = line.split()
        word = arr[0].split('.')[0]
        embedding =[]
        if word not in gloveDict:
            print('{} doesnt exist'.format(word))
            rMin, rMax = -6/np.sqrt(300), 6/np.sqrt(300)
            embedding = np.random.uniform(rMin, rMax, (1,300))[0]
        else:
            embedding = list(map(float, gloveDict[word]))

        eee = ''
        for elem in embedding:
            eee += '{} '.format(elem)
        print(eee, file=outputHandler)

    

