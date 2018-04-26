import codecs
import numpy as np
import string
from string import punctuation
from nltk.stem import WordNetLemmatizer

def load_word2vec():
    word2vec = {}

    print "==> loading 300d word2vec"
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
    f=open('/save/wenpeng/datasets/word2vec_words_300d.txt', 'r')#glove.6B.300d.txt, word2vec_words_300d.txt, glove.840B.300d.txt
    for line in f:
        l = line.split()
        word2vec[l[0]] = map(float, l[1:])

    print "==> word2vec is loaded"

    return word2vec

def load_word2vec_to_init(rand_values, ivocab, word2vec):
    fail=0
    for id, word in ivocab.iteritems():
        emb=word2vec.get(word)
        if emb is not None:
            rand_values[id]=np.array(emb)
        else:
#             print word
            fail+=1
    print '==> use word2vec initialization over...fail ', fail
    return rand_values

def transfer_wordlist_2_idlist_with_maxlen(token_list, vocab_map, maxlen):
    '''
    From such as ['i', 'love', 'Munich'] to idlist [23, 129, 34], if maxlen is 5, then pad two zero in the left side, becoming [0, 0, 23, 129, 34]
    '''
    idlist=[]
    for word in token_list:

        id=vocab_map.get(word)
        if id is None: # if word was not in the vocabulary
            id=len(vocab_map)+1  # id of true words starts from 1, leaving 0 to "pad id"
            vocab_map[word]=id
        idlist.append(id)

    mask_list=[1.0]*len(idlist) # mask is used to indicate each word is a true word or a pad word
    pad_size=maxlen-len(idlist)
    if pad_size>0:
        idlist=[0]*pad_size+idlist
        mask_list=[0.0]*pad_size+mask_list
    else: # if actual sentence len is longer than the maxlen, truncate
        idlist=idlist[:maxlen]
        mask_list=mask_list[:maxlen]
    return idlist, mask_list


def clean_text_to_wordlist(text):
    #remove punctuation
    clean_text = ''.join(c for c in text if c not in punctuation)
    wordlist =  clean_text.split()
    wordnet_lemmatizer = WordNetLemmatizer()
    return [wordnet_lemmatizer.lemmatize(word) for word in wordlist]


def load_reliefweb_dataset(maxlen=40):
    root="/save/wenpeng/datasets/LORELEI/"
    files=['ReliefWeb.train.balanced.txt', 'ReliefWeb.test.balanced.txt', 'ReliefWeb.test.balanced.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences=[]
    all_masks=[]
    all_labels=[]
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents=[]
        sents_masks=[]
        labels=[]
        readfile=codecs.open(root+files[i], 'r', 'utf-8')
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            label=int(parts[0])  # keep label be 0 or 1
            sentence_wordlist=clean_text_to_wordlist(parts[2].strip())

            labels.append(label)
            sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
            sents.append(sent_idlist)
            sents_masks.append(sent_masklist)
        all_sentences.append(sents)
        all_masks.append(sents_masks)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels)
    print 'dataset loaded over, totally ', len(word2id), 'words'
    return all_sentences, all_masks, all_labels, word2id


def load_BBN_dataset(maxlen=40):
    root="/save/wenpeng/datasets/LORELEI/SF-BBN-Mark-split/"
    files=['train.mark.12classes.txt', 'dev.mark.12classes.txt', 'test.mark.12classes.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences=[]
    all_masks=[]
    all_labels=[]
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents=[]
        sents_masks=[]
        labels=[]
        readfile=codecs.open(root+files[i], 'r', 'utf-8')
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            label=int(parts[0])  # keep label be 0 or 1
            sentence_wordlist=parts[2].strip().split()#clean_text_to_wordlist(parts[2].strip())

            labels.append(label)
            sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
            sents.append(sent_idlist)
            sents_masks.append(sent_masklist)
        all_sentences.append(sents)
        all_masks.append(sents_masks)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels)
    print 'dataset loaded over, totally ', len(word2id), 'words'
    return all_sentences, all_masks, all_labels, word2id



def load_BBN_multi_labels_dataset(maxlen=40):
    root="/save/wenpeng/datasets/LORELEI/SF-BBN-Mark-split/"
    files=['train.mark.multi.12labels.txt', 'dev.mark.multi.12labels.txt', 'test.mark.multi.12labels.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sentences=[]
    all_masks=[]
    all_labels=[]
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents=[]
        sents_masks=[]
        labels=[]
        readfile=codecs.open(root+files[i], 'r', 'utf-8')
        for line in readfile:
            parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            label=[0]*12
            for label_id in parts[0].strip().split():  # keep label be 0 or 1
                label[int(label_id)] =1
            sentence_wordlist=parts[2].strip().split()#clean_text_to_wordlist(parts[2].strip())

            labels.append(label)
            sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
            sents.append(sent_idlist)
            sents_masks.append(sent_masklist)
        all_sentences.append(sents)
        all_masks.append(sents_masks)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels)
    print 'dataset loaded over, totally ', len(word2id), 'words'
    return all_sentences, all_masks, all_labels, word2id
