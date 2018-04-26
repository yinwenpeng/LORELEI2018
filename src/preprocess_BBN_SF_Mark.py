import xml.etree.ElementTree as ET
import codecs

'''
set(['crimeviolence', 'med', 'search', 'food', 'out-of-domain', 'infra', 'water', 'shelter',
'regimechange', 'evac', 'terrorism', 'utils'])
'''
def split_bbn():

    root_path = '/save/wenpeng/datasets/LORELEI/SF-BBN-Mark-split/'
    topic2filename = {}
    raw_file_names = ['TRAIN', 'DEV', 'TEST']
    id2filename = {0: 'train', 1: 'dev', 2: 'test'}
    for i in range(3):
        readfile = open(root_path+raw_file_names[i], 'r')

        for line in readfile:
            topic = line.strip()
            topic2filename[topic] = id2filename.get(i)
        readfile.close()
    #split raw dataset
    type2label_id = {'crimeviolence':0, 'med':1, 'search':2, 'food':3, 'out-of-domain':4, 'infra':5, 'water':6, 'shelter':7,
    'regimechange':8, 'evac':9, 'terrorism':10, 'utils':11}
    tree = ET.parse(root_path+'LEIDOS_HADR_SF.xml')
    root = tree.getroot()
    write_train = codecs.open(root_path+'train.mark.12classes.txt', 'w', 'utf-8')
    write_dev = codecs.open(root_path+'dev.mark.12classes.txt', 'w', 'utf-8')
    write_test = codecs.open(root_path+'test.mark.12classes.txt', 'w', 'utf-8')

    type_set = set()
    for block in root.findall('SEGMENT'):
        sent = block.find('TOKENIZED_EXAMPLE').text
        sf_type = set(block.find('SF_TYPE').text.split())
        topic_ex = block.find('GUID').text
        type_set|=sf_type
        to_file = topic2filename.get(topic_ex)
        if to_file is None:
            print 'to_file is None:'
            exit(0)
        if to_file == 'train':
            for subtype in sf_type:
                write_train.write(str(type2label_id.get(subtype))+'\t'+subtype+'\t'+sent+'\n')
        elif to_file == 'dev':
            for subtype in sf_type:
                write_dev.write(str(type2label_id.get(subtype))+'\t'+subtype+'\t'+sent+'\n')
        else:
            for subtype in sf_type:
                write_test.write(str(type2label_id.get(subtype))+'\t'+subtype+'\t'+sent+'\n')
    write_train.close()
    write_dev.close()
    write_test.close()
    print 'all done'

def split_bbn_multi_label():

    root_path = '/save/wenpeng/datasets/LORELEI/SF-BBN-Mark-split/'
    topic2filename = {}
    raw_file_names = ['TRAIN', 'DEV', 'TEST']
    id2filename = {0: 'train', 1: 'dev', 2: 'test'}
    for i in range(3):
        readfile = open(root_path+raw_file_names[i], 'r')

        for line in readfile:
            topic = line.strip()
            topic2filename[topic] = id2filename.get(i)
        readfile.close()
    #split raw dataset
    type2label_id = {'crimeviolence':0, 'med':1, 'search':2, 'food':3, 'out-of-domain':4, 'infra':5, 'water':6, 'shelter':7,
    'regimechange':8, 'evac':9, 'terrorism':10, 'utils':11}
    tree = ET.parse(root_path+'LEIDOS_HADR_SF.xml')
    root = tree.getroot()
    write_train = codecs.open(root_path+'train.mark.multi.12labels.txt', 'w', 'utf-8')
    write_dev = codecs.open(root_path+'dev.mark.multi.12labels.txt', 'w', 'utf-8')
    write_test = codecs.open(root_path+'test.mark.multi.12labels.txt', 'w', 'utf-8')

    for block in root.findall('SEGMENT'):
        sent = block.find('TOKENIZED_EXAMPLE').text
        sf_type = block.find('SF_TYPE').text.split()  # a list
        sf_type_idlist = [str(type2label_id.get(typ)) for typ in sf_type]
        topic_ex = block.find('GUID').text

        to_file = topic2filename.get(topic_ex)
        if to_file is None:
            print 'to_file is None:'
            exit(0)
        if to_file == 'train':
            write_train.write(' '.join(sf_type_idlist)+'\t'+' '.join(sf_type)+'\t'+sent+'\n')
        elif to_file == 'dev':
            write_dev.write(' '.join(sf_type_idlist)+'\t'+' '.join(sf_type)+'\t'+sent+'\n')
        else:
            write_test.write(' '.join(sf_type_idlist)+'\t'+' '.join(sf_type)+'\t'+sent+'\n')
    write_train.close()
    write_dev.close()
    write_test.close()
    print 'all done'


if __name__ == '__main__':
    # split_bbn()
    split_bbn_multi_label()
