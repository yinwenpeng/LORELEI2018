import codecs


root = '/save/wenpeng/datasets/LORELEI/multi-lingual-emb/'

def load_il5_dictionary():
    #il5phrase to english word
    readfile = codecs.open(root+'il5-engligh.dictionary.txt', 'r', 'utf-8')
    il5phrase_2_enWord={}
    for line in readfile:
        parts=line.strip().split('\t')
        if len(parts)!=2:
            print line, parts[0], parts[1]
            exit(0)
        else:
            il5phrase_2_enWord[parts[1]] = parts[0]
    readfile.close()
    print 'load il5 vocab size: ', len(il5phrase_2_enWord)
    return il5phrase_2_enWord

def load_il6_dictionary():
    #il6phrase to english word
    readfile = codecs.open(root+'il6-engligh.dictionary.txt', 'r', 'utf-8')
    il6phrase_2_enPhrase={}
    for line in readfile:
        parts=line.strip().split('\t')
        if len(parts)!=11:
            print line, parts[0], parts[5]
            exit(0)
        else:
            il6phrase_2_enPhrase[parts[0]] = parts[5]
    readfile.close()
    print 'load il6 vocab size: ', len(il6phrase_2_enPhrase)
    return il6phrase_2_enPhrase

def load_trainingdata_il5():
    readfile = codecs.open(root+'il5-engligh.dictionary.txt', 'r', 'utf-8')
    il5_word2id={}
    english_word2id={}
    source_ids = []
    target_ids = []
    for line in readfile:
        parts=line.strip().split('\t')
        if len(parts)!=2:
            print line, parts[0], parts[1]
            exit(0)
        else:
            il5_phrase = parts[1]
            en_word = parts[0]
            il5phrase_2_enWord[parts[1]] = parts[0]


def unify_moni_linguals():
    '''
    first load tig2en, oro2en dictionary
    '''




    en_file = root+'wiki.en.vec'
    tigrinya_file = root+'mono-lingual-il5-xinli.vec'
    oromo_file = root+'mono-lingual-il6-xinli.vec'
    language_str_id = ['english_', 'tigrinya_', 'oromo_']
    word2id={}
    id2en = {}
    id2tig={}
    id2oro={}
