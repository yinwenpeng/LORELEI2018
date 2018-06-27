import codecs
import json
from pprint import pprint
import nltk
from collections import defaultdict

label2id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2, 'water':7, 'shelter':5,
'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}
def denoise_text(text):
    wordlist = []
    for word in text.strip().split():
        if len(word) < 15:
            wordlist.append(word)
    text =  ' '.join(wordlist)
    tokenized_text = nltk.word_tokenize(text)
    return ' '.join(tokenized_text)


def subset_reliefweb(label_size):
    readfile = codecs.open('/save/wenpeng/datasets/LORELEI/ReliefWeb_id_label_text.txt', 'r', 'utf-8')
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/ReliefWeb_subset_id_label_text.txt', 'w', 'utf-8')
    co=0
    for line in readfile:
        parts =  line.strip().split('\t')
        if len(parts[0].split())>=label_size:
            writefile.write(line.strip()+'\n')
            co+=1
    writefile.close()
    readfile.close()
    print 'subset reliefweb over with size:', co



def reformat_multilabel_reliefweb_json2txt():
    relief_data = json.load(codecs.open('/save/wenpeng/datasets/LORELEI/multilabel-relief.json', 'r', 'utf-8'))
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/ReliefWeb_id_label_text.txt', 'w', 'utf-8')
    size = len(relief_data)
    # label2id={}
    # id2amout=defaultdict(int)
    co=0
    multi_label_size = 0
    for i in range(size):
        label_list = relief_data[i]['labels']   # we found only one label per text
        if len(label_list) > 1:
            multi_label_size+=1
        text = denoise_text(relief_data[i]['body'])
        # print label, label2id
        ids = []
        for label in label_list:
            id = label2id.get(label)
            if id is None:
                print 'id',  id
                exit(0)
            ids.append(id)

        writefile.write(' '.join(map(str,ids))+'\t'+' '.join(label_list)+'\t'+text+'\n')
        co+=1
        if co % 1000 == 0:
            print co, '/', size
    print 'all done, multi_label_size:', multi_label_size
    writefile.close()
def reformat_reliefweb_json2txt():
    relief_data = json.load(codecs.open('/save/wenpeng/datasets/LORELEI/corpus-11.json', 'r', 'utf-8'))
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/ReliefWeb_id_label_text.txt', 'w', 'utf-8')
    size = len(relief_data)
    # label2id={}
    # id2amout=defaultdict(int)
    co=0
    for i in range(size):
        label_list = relief_data[i]['labels']   # we found only one label per text
        label = label_list[0]
        text = denoise_text(relief_data[i]['body'])
        # print label, label2id
        id = label2id.get(label)
        if id is None:
            print 'id',  id
            exit(0)
        writefile.write(str(id)+'\t'+label+'\t'+text+'\n')
        co+=1
        if co % 1000 == 0:
            print co, '/', size
    print 'all done'
    writefile.close()

def split_reliefweb_train_test():
    '''
    {0: 664, 1: 4042, 2: 736, 3: 3854, 4: 1149, 5: 1133, 6: 1512, 7: 633}) 13723
    {u'med': 3, u'search': 4, u'food': 1, u'utils': 6, u'infra': 2, u'water': 7, u'shelter': 5, u'evac': 0}
    '''
    readfile = codecs.open('/save/wenpeng/datasets/LORELEI/ReliefWeb_id_label_text.txt', 'r', 'utf-8')
    write_train = codecs.open('/save/wenpeng/datasets/LORELEI/ReliefWeb.train.txt', 'w', 'utf-8')
    write_test = codecs.open('/save/wenpeng/datasets/LORELEI/ReliefWeb.test.txt', 'w', 'utf-8')
    id2testsize={'0': 64, '1':542, '2':136, '3':554, '4':149,'5':133, '6':512, '7':33}
    id2testsize_dy = defaultdict(int)
    for line in readfile:
        id = line.strip().split('\t')[0]
        testsize_of_id = id2testsize_dy.get(id,0)
        if testsize_of_id < id2testsize.get(id):# 450*8 = 3600
            write_test.write(line.strip()+'\n')
            id2testsize_dy[id]+=1
        else:
            write_train.write(line.strip()+'\n')
    write_train.close()
    write_test.close()
    readfile.close()
    print 'all done, test size:', sum(id2testsize.values())

def split_reliefweb_train_test_balanced():
    '''
    {0: 664, 1: 4042, 2: 736, 3: 3854, 4: 1149, 5: 1133, 6: 1512, 7: 633}) 13723
    '''
    readfile = codecs.open('/save/wenpeng/datasets/LORELEI/ReliefWeb_id_label_text.txt', 'r', 'utf-8')
    write_train = codecs.open('/save/wenpeng/datasets/LORELEI/ReliefWeb.train.balanced.txt', 'w', 'utf-8')
    write_test = codecs.open('/save/wenpeng/datasets/LORELEI/ReliefWeb.test.balanced.txt', 'w', 'utf-8')
    id2testsize = defaultdict(int)
    for line in readfile:
        id = line.strip().split('\t')[0]
        testsize_of_id = id2testsize.get(id,0)
        if testsize_of_id < 450:# 450*8 = 3600
            write_test.write(line.strip()+'\n')
            id2testsize[id]+=1
        else:
            write_train.write(line.strip()+'\n')
    write_train.close()
    write_test.close()
    readfile.close()
    print 'all done, test size:', sum(id2testsize.values())







if __name__ == '__main__':
    # reformat_reliefweb_json2txt()
    # split_reliefweb_train_test()
    # split_reliefweb_train_test_balanced()
    # reformat_multilabel_reliefweb_json2txt()
    subset_reliefweb(3)
