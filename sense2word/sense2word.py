import numpy as np
from nltk.corpus import wordnet as wn

def read_sense_embeddings(file_path):
    senseE = {}
    with open(file_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            senseE[tokens[0]] = np.asarray(tokens[1:], dtype = float)
    return senseE

def write_word_embeddings(file_path, wordE):
    with open(file_path, "w") as f:
        for word in sorted(wordE.keys()):
            f.write("%s %s\n" % (word, " ".join(map(str, wordE[word]))))

def read_lexicon(file_path = "../glove.6B.50d.txt"):
    lexicon = []
    with open(file_path, "r") as f:
        for line in f:
            word = line.strip().split()[0]
            lexicon.append(word)
    return lexicon
    
def make_word_embeddings(senseE, lexicon, method = "averaged"):
    wordE = {}
    d = senseE[senseE.keys()[0]].shape
    for word in lexicon:
        v = np.zeros(d)
        try:
            synsets = wn.synsets(word)
        except:
            print "Encode error", word
            continue
        #if word != "apple":
            #continue
        if len(synsets) == 0:
            continue
        if method == "averaged":
            for sense in synsets:
                if sense.name() in senseE:
                    v += senseE[sense.name()]
            v /= len(synsets) 
        elif method == "expected":
            total = 0.0
            for sense in synsets:
                cnt = sum([l.count() for l in sense.lemmas()])
                v += senseE[sense.name()]*cnt
                total += cnt
            v /= total
        wordE[word] = v
        if max(v) == 0:
            del wordE[word]
    return wordE

if __name__ == "__main__":
    senseE = read_sense_embeddings("./e.small")
    lexicon = read_lexicon()
    wordE = make_word_embeddings(senseE, lexicon)
    write_word_embeddings("./smallE.txt", wordE)
