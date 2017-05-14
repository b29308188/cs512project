import numpy as np
from nltk.corpus import wordnet as wn

def read_sense_embeddings(file_path):
    senseE = {}
    with open(file_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) <= 2:
                continue
            senseE[tokens[0]] = np.asarray(tokens[1:], dtype = float)
    return senseE

def write_word_embeddings(file_path, wordE):
    with open(file_path, "w") as f:
        f.write("%d %d\n" % (len(wordE), len(wordE[wordE.keys()[0]])) )
        for word in sorted(wordE.keys()):
            f.write("%s %s\n" % (word, " ".join(map(str, wordE[word]))))

def read_lexicon(file_path = "../glove.6B.300d.txt"):
    lexicon = {}
    with open(file_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            lexicon[tokens[0]] = np.asarray(tokens[1:], dtype = float)
    return lexicon
    
def make_word_embeddings(senseE, lexicon, method = "averaged"):
    wordE = {}
    d = senseE[senseE.keys()[0]].shape
    for word in lexicon:
        v = np.zeros(d)
        try:
            synsets = wn.synsets(word)
        except:
            #print "Encode error", word
            continue
        #if word != "apple":
            #continue
        if len(synsets) == 0:
            continue
            #wordE[word] = lexicon[word]
        else:
            if method == "averaged":
                for sense in synsets:
                    if sense.name() in senseE:
                        v += senseE[sense.name()]
                v /= len(synsets) 
            elif method == "weighted":
                total = 0.0
                for sense in synsets:
                    cnt = sum([l.count() for l in sense.lemmas()])
                    if sense.name() in senseE:
                        v += senseE[sense.name()]*cnt
                        total += cnt
                if total > 0:
                    v /= total
            if max(v) > 0 or min(v) < 0: 
                wordE[word] = v
            else:
                continue
                #wordE[word] = lexicon[word]
    print len(wordE)
    return wordE

if __name__ == "__main__":
    senseE = read_sense_embeddings("./glove_reg_tranR-2.txt")
    lexicon = read_lexicon()
    #wordE = make_word_embeddings(senseE, lexicon, method = "weighted")
    wordE = make_word_embeddings(senseE, lexicon, method = "averaged")
    write_word_embeddings("./glove_reg_tranR_word_noglove-2.txt", wordE)
