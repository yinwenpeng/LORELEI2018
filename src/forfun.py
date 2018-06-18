import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time
import math
import codecs
import numpy as np


def load_fasttext_word2vec_given_file(filepath, dim):
    word2vec = {}

    print filepath, "==> loading 300d word2vec"
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
    f=codecs.open(filepath, 'r', 'utf-8', errors='ignore')#glove.6B.300d.txt, word2vec_words_300d.txt, glove.840B.300d.txt
    line_co = 0
    for line in f:
        l = line.split()
        # print l
        if len(l)==dim+1:
            value_list = map(float, l[1:])
            # norm = LA.norm(np.asarray(value_list))
            # word2vec[l[0]] = [value/norm for value in value_list]
            word2vec[l[0]] = value_list
            line_co+=1
            # if line_co % 100:
            print 'line_co:', line_co, l[0]
            # if line_co > 100:
            #     break

    print "==> word2vec is loaded over"

    return word2vec


if __name__ == '__main__':
    load_fasttext_word2vec_given_file('/save/wenpeng/datasets/LORELEI/multi-lingual-emb/il5_300d_word2vec.txt', 300)
