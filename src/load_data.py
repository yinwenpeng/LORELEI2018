import codecs
import numpy as np
import string
from string import punctuation
from nltk.stem import WordNetLemmatizer
from numpy import linalg as LA

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
            # if line_co % 500000:
            #     print 'line_co:', line_co
            if line_co > 1000000:
                break

    print "==> word2vec is loaded over"

    return word2vec

def load_fasttext_multiple_word2vec_given_file(filepath_list, dim):
    word2vec = {}
    for fil in filepath_list:
        print fil, "==> loading 300d word2vec"
    #     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
        f=codecs.open(fil, 'r', 'utf-8',errors='ignore')#glove.6B.300d.txt, word2vec_words_300d.txt, glove.840B.300d.txt
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
                # if line_co % 500000:
                #     print 'line_co:', line_co
                if line_co > 1000000:
                    break

        print "==> word2vec is loaded over"

    return word2vec

def load_word2vec_to_init(rand_values, ivocab, word2vec):
    fail=0
    for id, word in ivocab.iteritems():
        emb=word2vec.get(word)
        if emb is not None:
            rand_values[id]=np.array(emb)
        else:
            # print word
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

def load_trainingdata_il6(root, english_vocab, il5_vocab, max_il5_phrase_len, max_en_phrase_len):
    readfile = codecs.open(root+'il6-engligh.dictionary.txt', 'r', 'utf-8')
    il5_word2id={}
    english_word2id={}
    source_ids = []
    source_masks = []
    target_ids = []
    target_masks = []
    for line in readfile:
        parts=line.strip().split('\t')
        if len(parts)!=11:
            print line, parts[0], parts[1]
            exit(0)
        else:
            # sent_ids =


            il5_phrase = parts[0]
            en_phrase  = parts[5]


            il5_wordlist = [word for word in il5_phrase.split() if word in il5_vocab]
            if len(il5_wordlist) < 1:
                continue

            source_idlist, source_masklist = transfer_wordlist_2_idlist_with_maxlen(il5_wordlist, il5_word2id, max_il5_phrase_len)
            target_idlist, target_masklist = transfer_wordlist_2_idlist_with_maxlen(en_phrase.split(), english_word2id, max_en_phrase_len)

            source_ids.append(source_idlist)
            source_masks.append(source_masklist)
            target_ids.append(target_idlist)
            target_masks.append(target_masklist)
    readfile.close()
    print 'load il6 training data over, size: ', len(target_ids)
    return source_ids, source_masks, target_ids,target_masks, il5_word2id, english_word2id

def load_trainingdata_il5(root, english_vocab, il5_vocab, max_il5_phrase_len):
    readfile = codecs.open(root+'il5-engligh.dictionary.txt', 'r', 'utf-8')
    il5_word2id={}
    english_word2id={}
    source_ids = []
    source_masks = []
    target_ids = []
    for line in readfile:
        parts=line.strip().split('\t')
        if len(parts)!=2:
            print line, parts[0], parts[1]
            exit(0)
        else:
            # sent_ids =


            il5_phrase = parts[1]
            en_word = parts[0]


            il5_wordlist = [word for word in il5_phrase.split() if word in il5_vocab]
            if len(il5_wordlist) < 1:
                continue

            sent_idlist, sent_masklist = transfer_wordlist_2_idlist_with_maxlen(il5_wordlist, il5_word2id, max_il5_phrase_len)
            if en_word in english_vocab:
                id_en_word = english_word2id.get(en_word)
                if id_en_word is None:
                    id_en_word = len(english_word2id)
                    english_word2id[en_word] = id_en_word

                source_ids.append(sent_idlist)
                source_masks.append(sent_masklist)
                target_ids.append(id_en_word)
    readfile.close()
    print 'load il5 training data over, size: ', len(target_ids)
    return source_ids, source_masks, target_ids, il5_word2id, english_word2id

def clean_text_to_wordlist(text):
    #remove punctuation
    clean_text = ''.join(c for c in text if c not in punctuation)
    wordlist =  clean_text.split()
    wordnet_lemmatizer = WordNetLemmatizer()
    return [wordnet_lemmatizer.lemmatize(word) for word in wordlist]

def load_reliefweb_il5_12_multilabel(maxlen):
    root="/save/wenpeng/datasets/LORELEI/"
    # files=['ReliefWeb.train.balanced.txt', 'ReliefWeb.test.balanced.txt', 'il5_labeled_as_training_seg_level.txt']
    files = ['ReliefWeb_id_label_text.txt','il5_labeled_as_training_seg_level.txt','il5_labeled_as_training_seg_level.txt']
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
            # label=int(parts[0])  # keep label be 0 or 1
            label_vec=[0]*12
            for label_id in parts[0].strip().split():  # keep label be 0 or 1
                label_vec[int(label_id)] =1

            sentence_wordlist=clean_text_to_wordlist(parts[2].strip())

            labels.append(label_vec)
            sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sentence_wordlist, word2id, maxlen)
            sents.append(sent_idlist)
            sents_masks.append(sent_masklist)
        all_sentences.append(sents)
        all_masks.append(sents_masks)
        all_labels.append(labels)
        print '\t\t\t size:', len(labels)
    print 'dataset loaded over, totally ', len(word2id), 'words'
    return all_sentences, all_masks, all_labels, word2id

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



def load_reliefweb_E30_dataset(maxlen=40):
    root="/save/wenpeng/datasets/LORELEI/"
    files=['ReliefWeb.train.balanced.txt', 'ReliefWeb.test.balanced.txt', 'translated2017/E30_id_label_segment.txt']
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
            # label=int(parts[0])  # keep label be 0 or 1
            label=[0]*8
            for label_id in parts[0].strip().split():  # keep label be 0 or 1
                label[int(label_id)] =1
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

def load_SF_type_descriptions(word2id, type_size, describ_max_len):
    # type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2, 'water':7, 'shelter':5,
    # 'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}
    type2des = {0:'evacuation evacuate landslide flood volcano earthquake',
                        1:'food hunger starvation starve',
                        2:'infrastructure  damage house collapse water pipe burst no electricity road',
                        3: 'medical assistance sick flu dysentery patient insufficiency',
                        4: 'search house collapse  person  missing  buried',
                        5: 'shelter house collapse homeless',
                        6: 'utilities energy sanitation electricity',
                        7: 'water food hunger starvation starve water pollution',
                        8: 'crime violence robbery snoring looting burning plunder shooting blow up explode attack arrest kill shot police incident',
                        9: 'none nothing',
                        10: 'regime, change coup overthrow subversion resign subvert turn over rebel army',
                        11: 'terrorism blow up explode shooting suicide attack terrorist conspiracy explosion terror bombing bomb isis'
                        }
    # type2des = {0:'nothing',
    #                     1:'nothing',
    #                     2:'nothing',
    #                     3: 'nothing',
    #                     4: 'nothing',
    #                     5: 'nothing',
    #                     6: 'nothing',
    #                     7: 'nothing',
    #                     8: 'nothing',
    #                     9: 'none nothing',
    #                     10: 'nothing',
    #                     11: 'nothing'
    #                     }
    label_sent = []
    label_mask = []
    for i in range(type_size):
        sent_str = type2des.get(i)
        sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sent_str.split(), word2id, describ_max_len)
        label_sent.append(sent_idlist)
        label_mask.append(sent_masklist)
    return label_sent, label_mask



def load_il6_with_BBN(maxlen=40):
    root="/save/wenpeng/datasets/LORELEI/"
    files=['SF-BBN-Mark-split/full_BBN_multi.txt', 'SF-BBN-Mark-split/full_BBN_multi.txt', 'il6_labeled_as_training_seg_level.txt']
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

def load_BBN_multi_labels_dataset(maxlen=40):
    root="/save/wenpeng/datasets/LORELEI/"
    # files=['SF-BBN-Mark-split/train.mark.multi.12labels.txt', 'SF-BBN-Mark-split/dev.mark.multi.12labels.txt', 'il5_labeled_as_training_seg_level.txt']
    files=['SF-BBN-Mark-split/full_BBN_multi.txt', 'SF-BBN-Mark-split/full_BBN_multi.txt', 'il5_labeled_as_training_seg_level.txt']
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

def load_BBN_8_labels_testOn_E30_dataset(maxlen=40):
    root="/save/wenpeng/datasets/LORELEI/"
    files=['SF-BBN-Mark-split/train.mark.multi.12labels.txt', 'SF-BBN-Mark-split/dev.mark.multi.12labels.txt', 'translated2017/E30_id_label_segment.txt']
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
                label_id_int  = int(label_id)
                if label_id_int < 8:
                    label[label_id_int] =1
            if sum(label) == 0:
                continue
            else:
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
