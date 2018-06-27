import codecs
import json
from pprint import pprint
import nltk
from collections import defaultdict
import os
import xml.etree.ElementTree as ET




def valid_seg(doc_id, seg_id,doc2remain_seglist):
    if len(doc2remain_seglist) == 0: # means there is no filter
        return True
    if doc_id in doc2remain_seglist:
        seg_list = doc2remain_seglist.get(doc_id)
        for gold_seg_id in seg_list:

            if seg_id == gold_seg_id:
                return True
                break
    return False

def SF_annotated_docSet(filepath):
    docset=set()
    readfile = codecs.open(filepath, 'r', 'utf-8')
    co=0
    for line in readfile:
        if co > 0:
            parts=line.strip().split('\t')
            docset.add(parts[0])
        co+=1
    print 'SF annotated doc set loaded over, size:', len(docset)
    return docset

def find_entities_for_window(entity_pos_list, seg_start, seg_end):
    # print 'entity_pos_list, seg_start, seg_end:',entity_pos_list, seg_start, seg_end
    seg_start = int(seg_start)
    seg_end = int(seg_end)
    pos_list = []
    nearest_pos = ' '
    min_distance = 100000000
    for entity_pos in entity_pos_list:
        parts = entity_pos.split('-')
        start = int(parts[0])
        end = int(parts[1])
        if start >= seg_start and end <=seg_end:
            pos_list.append(entity_pos)
        else: # outside the segment
            if end < seg_start:
                distance = seg_start-end
            if start> seg_end:
                distance = start - seg_end
            if distance < min_distance:
                min_distance = distance
                nearest_pos = entity_pos
    '''
    if no entity detected in this seg, whether or not search the nearest entity
    '''
    # print 'pos_list:',pos_list
    # if len(pos_list) == 0:
    #     '''
    #     means we have to search the nearest entity for this segments
    #     '''
    #     assert nearest_pos != ' '
    #     pos_list.append(nearest_pos)
    return pos_list

def IL_into_test_filteredby_NER(ltf_path, docSet, writefile_path, docid2entity_pos_list, window):
    '''
    docid2entity_pos_list: doc_id: ['12-14', '34-67', ...]
    '''
    writefile_path=writefile_path+'_w'+str(window)+'.txt'
    writefile = codecs.open(writefile_path, 'w', 'utf-8')
    # write_wp_file = codecs.open('/save/wenpeng/datasets/LORELEI/wp_filelist.txt', 'w', 'utf-8')
    # for docc in docSet:
    #     write_wp_file.write(docc+'\tTRUE\n')
    # write_wp_file.close()
    vocab=set()
    re=0

    files= os.listdir(ltf_path)
    print 'folder file sizes: ', len(files)
    co=0
    for fil in files:
        f = ET.parse(ltf_path+fil)
        root = f.getroot()

        for doc in root.iter('DOC'):
            doc_id  = doc.attrib.get('id')
        if doc_id in docSet and doc_id in docid2entity_pos_list:
        # if doc_id in docid2entity_pos_list:

            list_docSegSent=[]
            for seg in root.iter('SEG'):
                sent_wordlist = []
                seg_id = seg.attrib.get('id')
                seg_start = seg.attrib.get('start_char')
                seg_end = seg.attrib.get('end_char')
                for word in seg.iter('TOKEN'):
                    word_str = word.text
                    if word_str.find('https') <0:
                        sent_wordlist.append(word_str)
                if len(sent_wordlist) > 0:
                    list_docSegSent.append((doc_id,seg_id, seg_start,seg_end, ' '.join(sent_wordlist)))
            #scan window to write
            if len(list_docSegSent) <=2*window+1:
                sent=''
                for ele in list_docSegSent:
                    sent+= ' '+ele[4]
                pos_list = docid2entity_pos_list.get(doc_id)
                writefile.write(doc_id+'\t'+list_docSegSent[0][1]+'\t'+sent.strip()+'\t'+' '.join(pos_list)+'\n')
            else:
                for i, trip in enumerate(list_docSegSent):
                    doc_idd = trip[0]
                    entity_pos_list = docid2entity_pos_list.get(doc_idd)
                    seg_idd = trip[1]
                    seg_start = trip[2]
                    seg_end  = trip[3]
                    print 'B seg_start, seg_end:',len(list_docSegSent), doc_idd, seg_start, seg_end
                    sent = trip[4]
                    for j in range(i-1, i-window-1, -1):
                        if j>=0:
                            sent = list_docSegSent[j][4]+' '+sent
                            seg_start = list_docSegSent[j][2]
                    for j in range(i+1, i+window+1):
                        if j < len(list_docSegSent):
                            sent=sent +' '+list_docSegSent[j][4]
                            seg_end = list_docSegSent[j][3]
                    print 'A seg_start, seg_end:',len(list_docSegSent), doc_idd, seg_start, seg_end
                    pos_list = find_entities_for_window(entity_pos_list, seg_start, seg_end)
                    if len(pos_list) > 0:
                        writefile.write(doc_idd+'\t'+seg_idd+'\t'+sent+'\t'+' '.join(pos_list)+'\n')
            co+=1
            if co % 1000 == 0:
                print 'generating standard test instances...', co
    writefile.close()

    print 'over'

def preprocess_NER_results():
    readfile = codecs.open('/save/wenpeng/datasets/LORELEI/Thai/NerResult/LDC2018E03_UpennEDL_2018-06-18-21-25.tab', 'r', 'utf-8')
    doc2entitylist=defaultdict(list)
    doc_set = set()
    valid_size = 0
    for line in readfile:
        parts = line.strip().split('\t')
        doc_id = parts[1][:parts[1].find('-')]
        doc_set.add(doc_id)
        span = parts[3][parts[3].find(':')+1:].split('-')
        start = int(span[0])
        end = int(span[1])
        type = parts[5]
        if type == 'GPE':
            entity_instance = {}
            entity_instance['start'] = start
            entity_instance['end'] = end
            doc2entitylist[doc_id].append(entity_instance)
            valid_size+=1
    readfile.close()
    print 'load ner results over, size:', valid_size, 'doc size:', len(doc_set)
    return doc2entitylist

def Thai_rich_info():
    folder = '/save/wenpeng/datasets/LORELEI/Thai/setE/'
    doc2seglist=defaultdict(list)
    files= os.listdir(folder)
    print 'folder file sizes: ', len(files)
    co=0
    for fil in files:
        f = ET.parse(folder+fil)
        root = f.getroot()
        for doc in root.iter('DOC'):
            doc_id  = doc.attrib.get('id')
        for seg in root.iter('SEG'):
            seg_instance = {}
            sent_wordlist = []
            seg_id = seg.attrib.get('id')
            seg_start = int(seg.attrib.get('start_char'))
            seg_end = int(seg.attrib.get('end_char'))
            for word in seg.iter('TOKEN'):
                word_str = word.text
                sent_wordlist.append(word_str)
            seg_instance['id'] = seg_id
            seg_instance['seg_start'] = seg_start
            seg_instance['seg_end'] = seg_end
            seg_instance['text'] = ' '.join(sent_wordlist)
            doc2seglist[doc_id].append(seg_instance)

            co+=1
            # if co % 1000 == 0:
            #     print 'co...', co
    print 'Thai seg boundary info loaded over, seg_instance size', co
    return doc2seglist

def filter_ner_results():
    doc2entitylist = preprocess_NER_results()
    doc2seglist = Thai_rich_info()
    size = 0
    doc2remain_seglist=defaultdict(list)
    for doc_id in doc2entitylist:
        if doc_id in doc2seglist:
            ent_list = doc2entitylist.get(doc_id)
            seg_list = doc2seglist.get(doc_id)
            for ent_instance in ent_list:
                start = ent_instance.get('start')
                end= ent_instance.get('end')
                for seg in seg_list:
                    seg_start = seg.get('seg_start')
                    seg_end = seg.get('seg_end')
                    seg_id = seg.get('id')
                    if start >=seg_start and end <=seg_end:
                        doc2remain_seglist[doc_id].append(seg_id)
                        size+=1
    print 'remain doc size:', len(doc2remain_seglist), ' seg size:', size
    return doc2remain_seglist
