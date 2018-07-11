import codecs
import json
from pprint import pprint
import nltk
from collections import defaultdict,Counter
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
            if (start<seg_start and end > seg_start and end<seg_end) or (start>seg_start and start < seg_end and end>seg_end):
                distance = 0
            if distance < min_distance:
                min_distance = distance
                nearest_pos = entity_pos
    '''
    if no entity detected in this seg, whether or not search the nearest entity
    '''
    # print 'pos_list:',pos_list
    if len(pos_list) == 0:
        '''
        means we have to search the nearest entity for this segments
        '''
        assert nearest_pos != ' '
        pos_list.append(nearest_pos)
    return pos_list

def IL_eng_into_test_filteredby_NER_2018(ltf_path, writefile_path, window):
    '''
    docid2entity_pos_list: doc_id: ['12-14-kbid', '34-67-kbid', ...]
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
        if True:#doc_id in docid2entity_pos_list:
        # if True:#doc_id in docid2entity_pos_list:

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
                # pos_list = docid2entity_pos_list.get(doc_id)
                writefile.write(doc_id+'\t'+list_docSegSent[0][1]+'\t'+sent.strip()+'\n')
                # writefile.write(doc_id+'\t'+list_docSegSent[0][1]+'\t'+sent.strip()+'\n')
            else:
                for i, trip in enumerate(list_docSegSent):
                    doc_idd = trip[0]
                    # entity_pos_list = docid2entity_pos_list.get(doc_idd)
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
                    # pos_list = find_entities_for_window(entity_pos_list, seg_start, seg_end)
                    # if len(pos_list) > 0:
                    writefile.write(doc_idd+'\t'+seg_idd+'\t'+sent+'\n')
                    # writefile.write(doc_idd+'\t'+seg_idd+'\t'+sent+'\n')
            co+=1
            if co % 1000 == 0:
                print 'generating standard test instances...', co
    writefile.close()

    print 'over'

def IL_into_test_filteredby_NER_2018(ltf_path, writefile_path, docid2entity_pos_list, window):
    '''
    docid2entity_pos_list: doc_id: ['12-14-kbid', '34-67-kbid', ...]
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
        if doc_id in docid2entity_pos_list:
        # if True:#doc_id in docid2entity_pos_list:

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
                # writefile.write(doc_id+'\t'+list_docSegSent[0][1]+'\t'+sent.strip()+'\n')
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
                    # writefile.write(doc_idd+'\t'+seg_idd+'\t'+sent+'\n')
            co+=1
            if co % 1000 == 0:
                print 'generating standard test instances...', co
    writefile.close()

    print 'over'


def load_BBN_MT_folder(mt_path):
    '''
    try tokenize
    '''
    files= os.listdir(mt_path)
    print 'BBN MT file sizes: ', len(files)
    co=0
    docid2sent_list = {}
    for fil in files:
        # print mt_path+fil
        f = ET.parse(mt_path+fil)
        root = f.getroot()
        for doc in root.iter('doc'):
            doc_id  = doc.attrib.get('docid')
        sent_list = []
        for sent_str in root.iter('seg'):
            raw_sent_str = sent_str.text
            tokenized_text = nltk.word_tokenize(raw_sent_str)
            sent_list.append(' '.join(tokenized_text))
        docid2sent_list[doc_id] = sent_list
    return docid2sent_list


def IL_into_test_withMT_filteredby_NER_2018(ltf_path, mt_path, writefile_path, docid2entity_pos_list, window):
    '''
    docid2entity_pos_list: doc_id: ['12-14-kbid', '34-67-kbid', ...]
    '''

    docid2mt_sent_list = load_BBN_MT_folder(mt_path)

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
        if doc_id in docid2entity_pos_list:
        # if True:#doc_id in docid2entity_pos_list:

            list_docSegSent=[]
            for seg in root.iter('SEG'):
                sent_wordlist = []
                seg_id = seg.attrib.get('id')
                seg_start = seg.attrib.get('start_char')
                seg_end = seg.attrib.get('end_char')
                for word in seg.iter('TOKEN'):
                    word_str = word.text
                    # if word_str.find('https') <0:
                    sent_wordlist.append(word_str)
                if len(sent_wordlist) > 0:
                    list_docSegSent.append((doc_id,seg_id, seg_start,seg_end, ' '.join(sent_wordlist)))
            #scan window to write
            if len(list_docSegSent) <=2*window+1:
                sent=''
                for ele in list_docSegSent:
                    sent+= ' '+ele[4]
                pos_list = docid2entity_pos_list.get(doc_id)
                '''
                MT
                '''
                mt_sent_list = docid2mt_sent_list.get(doc_id)
                if mt_sent_list is None:
                    mt_sent = sent
                else:
                    mt_sent = ' '.join(mt_sent_list)
                writefile.write(doc_id+'\t'+list_docSegSent[0][1]+'\t'+sent.strip()+'\t'+mt_sent.strip()+'\t'+' '.join(pos_list)+'\n')
                # writefile.write(doc_id+'\t'+list_docSegSent[0][1]+'\t'+sent.strip()+'\n')
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
                    '''
                    MT
                    '''
                    mt_sent_list = docid2mt_sent_list.get(doc_id)
                    if mt_sent_list is None:
                        mt_sent = sent
                    else:
                        # print doc_id,len(mt_sent_list), len(list_docSegSent)
                        assert len(mt_sent_list) == len(list_docSegSent)
                        mt_sent = mt_sent_list[i]
                        for j in range(i-1, i-window-1, -1):
                            if j>=0:
                                mt_sent = mt_sent_list[j]+' '+mt_sent
                        for j in range(i+1, i+window+1):
                            if j < len(mt_sent_list):
                                mt_sent=mt_sent +' '+mt_sent_list[j]

                    print 'A seg_start, seg_end:',len(list_docSegSent), doc_idd, seg_start, seg_end
                    pos_list = find_entities_for_window(entity_pos_list, seg_start, seg_end)
                    if len(pos_list) > 0:
                        writefile.write(doc_idd+'\t'+seg_idd+'\t'+sent.strip()+'\t'+mt_sent.strip()+'\t'+' '.join(pos_list)+'\n')
                    # writefile.write(doc_idd+'\t'+seg_idd+'\t'+sent+'\n')
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


def get_need_other_fields(matrix):
    #matrix (4,4)
    # other_field2index = {'current':0,'not_current':1, 'sufficient':0,'insufficient':1,'True':0,'False':1}
    # other_fields=[2]*4 #need_status, issue_status, need_relief, need_urgency, defalse "2" denotes no label
    fields_size = len(matrix)
    assert fields_size ==4
    if matrix[0][0]>matrix[0][1] and matrix[0][0]>matrix[0][2]:
        status = 'current'
    elif matrix[0][1]>matrix[0][0] and matrix[0][1]>matrix[0][2]:
        status = 'future'
    elif matrix[0][2]>matrix[0][0] and matrix[0][2]>matrix[0][1]:
        status = 'past'
    if matrix[2][0]>matrix[2][1]:
        relief = 'sufficient'
    else:
        relief = 'insufficient'
    if matrix[3][0]>matrix[3][1]:
        urgency = True
    else:
        urgency = False
    return    status,  relief,urgency

def get_issue_other_fields(matrix):
    #matrix (4,3)
    # other_field2index = {'current':0,'not_current':1, 'sufficient':0,'insufficient':1,'True':0,'False':1}
    # other_fields=[2]*4 #need_status, issue_status, need_relief, need_urgency, defalse "2" denotes no label
    fields_size = len(matrix)
    assert fields_size ==4
    if matrix[1][0]>matrix[1][1]:
        status = 'current'
    else:
        status = 'not_current'
    if matrix[3][0]>matrix[3][1]:
        urgency = True
    else:
        urgency = False
    return    status, urgency

def generate_2018_official_output_english(lines, output_file_path, pred_types, pred_confs, pred_others, min_mean_frame):
    #pred_others (#instance, 4, 3)
    # thai_root = '/save/wenpeng/datasets/LORELEI/Thai/'
    instance_size = len(pred_types)
    type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2,
    'water':7, 'shelter':5, 'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}

    id2type = {y:x for x,y in type2label_id.iteritems()}

    output_dict_list = []
    assert instance_size == len(pred_others)
    assert instance_size == len(pred_confs)
    assert instance_size == len(lines)
    print 'seg size to pred: ', instance_size, 'full file size:', len(lines)
    # assert instance_size == len(lines)


    # pred_needs = pred_types[:,:8]
    # pred_issues = np.concatenate([pred_types[:,8:9], pred_types[:, 10:]),axis=1)  #(all, 3)

    #needs
    for i in range(instance_size):
        # print 'lines[i]:', lines[i].split('\t')
        # entity_pos_list = ['10-13-GPE']# lines[i].split('\t')[3].split()
        pred_vec = list(pred_types[i])
        text_parts = lines[i].split('\t')
        doc_id = text_parts[0]
        seg_id = text_parts[1]
        # entity_pos_list = text_parts[3].split() #116-123-6252001 125-130-49518 198-203-49518
        for x, y in enumerate(pred_vec):
            if y == 1:
                if x < 8: # is a need type
                    # for entity_pos in entity_pos_list:
                        # kb_id = entity_pos.split('-')[2]
                    new_dict={}
                    new_dict['DocumentID'] = doc_id
                    hit_need_type = id2type.get(x)
                    new_dict['Type'] = hit_need_type
                    new_dict['Place_KB_ID'] = 'TBD'
                    status,  relief,urgency = get_need_other_fields(pred_others[i])
                    new_dict['Status'] = status
                    new_dict['Confidence'] = float(pred_confs[i][x])
                    new_dict['Justification_ID'] = seg_id
                    new_dict['Resolution'] = relief
                    new_dict['Urgent'] = urgency
                    if new_dict.get('Confidence') > 0.4:
                        output_dict_list.append(new_dict)

                elif x ==8 or x > 9: # is issue
                    # for entity_pos in entity_pos_list:
                        # kb_id = entity_pos.split('-')[2]
                    new_dict={}
                    new_dict['DocumentID'] = doc_id
                    hit_issue_type = id2type.get(x)
                    new_dict['Type'] = hit_issue_type
                    new_dict['Place_KB_ID'] = 'TBD'
                    # new_dict['Place_'] = 14.0
                    status, urgency = get_issue_other_fields(pred_others[i])
                    new_dict['Status'] = status
                    new_dict['Confidence'] = float(pred_confs[i][x])
                    new_dict['Justification_ID'] = seg_id
                    new_dict['Urgent'] = urgency
                    if new_dict.get('TypeConfidence') > 0.4:
                        output_dict_list.append(new_dict)


    refine_output_dict_list, ent_size = de_duplicate(output_dict_list)
    frame_size = len(refine_output_dict_list)
    mean_frame = frame_size*1.0/ent_size
    # if mean_frame < min_mean_frame:
    writefile = codecs.open(output_file_path ,'w', 'utf-8')
    json.dump(refine_output_dict_list, writefile)
    writefile.close()
    print 'official output succeed...Frame size:', frame_size, 'average:', mean_frame, 'ent_size:',ent_size
    return mean_frame

def generate_2018_official_output(lines, output_file_path, pred_types, pred_confs, pred_others, min_mean_frame):
    #pred_others (#instance, 4, 3)
    # thai_root = '/save/wenpeng/datasets/LORELEI/Thai/'
    instance_size = len(pred_types)
    type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2,
    'water':7, 'shelter':5, 'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}

    id2type = {y:x for x,y in type2label_id.iteritems()}

    output_dict_list = []
    assert instance_size == len(pred_others)
    assert instance_size == len(pred_confs)
    assert instance_size == len(lines)
    print 'seg size to pred: ', instance_size, 'full file size:', len(lines)
    # assert instance_size == len(lines)


    # pred_needs = pred_types[:,:8]
    # pred_issues = np.concatenate([pred_types[:,8:9], pred_types[:, 10:]),axis=1)  #(all, 3)

    #needs
    for i in range(instance_size):
        # print 'lines[i]:', lines[i].split('\t')
        # entity_pos_list = ['10-13-GPE']# lines[i].split('\t')[3].split()
        pred_vec = list(pred_types[i])
        text_parts = lines[i].split('\t')
        doc_id = text_parts[0]
        seg_id = text_parts[1]
        entity_pos_list = text_parts[3].split() #116-123-6252001 125-130-49518 198-203-49518
        for x, y in enumerate(pred_vec):
            if y == 1:
                if x < 8: # is a need type
                    for entity_pos in entity_pos_list:
                        kb_id = entity_pos.split('-')[2]
                        new_dict={}
                        new_dict['DocumentID'] = doc_id
                        hit_need_type = id2type.get(x)
                        new_dict['Type'] = hit_need_type
                        new_dict['Place_KB_ID'] = kb_id
                        status,  relief,urgency = get_need_other_fields(pred_others[i])
                        new_dict['Status'] = status
                        new_dict['Confidence'] = float(pred_confs[i][x])
                        new_dict['Justification_ID'] = seg_id
                        new_dict['Resolution'] = relief
                        new_dict['Urgent'] = urgency
                        if new_dict.get('Confidence') > 0.4:
                            output_dict_list.append(new_dict)

                elif x ==8 or x > 9: # is issue
                    for entity_pos in entity_pos_list:
                        kb_id = entity_pos.split('-')[2]
                        new_dict={}
                        new_dict['DocumentID'] = doc_id
                        hit_issue_type = id2type.get(x)
                        new_dict['Type'] = hit_issue_type
                        new_dict['Place_KB_ID'] = kb_id
                        # new_dict['Place_'] = 14.0
                        status, urgency = get_issue_other_fields(pred_others[i])
                        new_dict['Status'] = status
                        new_dict['Confidence'] = float(pred_confs[i][x])
                        new_dict['Justification_ID'] = seg_id
                        new_dict['Urgent'] = urgency
                        if new_dict.get('TypeConfidence') > 0.4:
                            output_dict_list.append(new_dict)


    refine_output_dict_list, ent_size = de_duplicate(output_dict_list)
    frame_size = len(refine_output_dict_list)
    mean_frame = frame_size*1.0/ent_size
    # if mean_frame < min_mean_frame:
    writefile = codecs.open(output_file_path ,'w', 'utf-8')
    json.dump(refine_output_dict_list, writefile)
    writefile.close()
    print 'official output succeed...Frame size:', frame_size, 'average:', mean_frame, 'ent_size:',ent_size
    return mean_frame

def best_seg_id(segs, window):
    #[segment-2, segment-7, ...]
    #window =2
    id_list = []
    for seg in segs:
        id_list.append(int(seg.split('-')[1]))
    min_seg = min(id_list)
    max_seg = max(id_list)
    extend_list = []
    for idd in id_list:
        for i in range(idd-window, idd+window+1):
            if i >= min_seg and i <=max_seg:
                extend_list.append(i)

    majority_id = majority_ele_in_list(extend_list)
    return 'segment-'+str(majority_id)


def de_duplicate(output_dict_list):
    need_type_set = set([ 'med','search','food','infra','water','shelter','evac','utils'])
    issue_type_set = set(['regimechange','crimeviolence','terrorism'])
    new_dict_list=[]
    key2dict_list = defaultdict(list)
    ent_set = set()
    for dic in output_dict_list:
        doc_id = dic.get('DocumentID')
        type = dic.get('Type')
        kb_id = dic.get('Place_KB_ID')
        key = (doc_id, type, kb_id)
        ent_set.add((doc_id, kb_id))
        key2dict_list[key].append(dic)
    for key, dict_list in key2dict_list.iteritems():
        #compute status, confidence
        doc_id = key[0]
        SF_type = key[1]
        kb_id = key[2]
        if dict_list[0].get('Type') in need_type_set:
            status=[]
            relief=[]
            urgency=[]
            conf = []
            segs = []

            for dic in dict_list:
                status.append(dic.get('Status'))
                relief.append(dic.get('Resolution'))
                urgency.append(dic.get('Urgent'))
                conf.append(dic.get('Confidence'))
                segs.append(dic.get('Justification_ID'))
            status = majority_ele_in_list(status)
            relief = majority_ele_in_list(relief)
            urgency = majority_ele_in_list(urgency)
            conf = max(conf)
            seg_id = best_seg_id(segs,2)

            new_dict={}
            new_dict['DocumentID'] = doc_id
            new_dict['Type'] = SF_type
            new_dict['Place_KB_ID'] =  kb_id
            new_dict['Status'] = status
            new_dict['Confidence'] = conf
            new_dict['Justification_ID'] = seg_id
            new_dict['Resolution'] = relief
            new_dict['Urgent'] = urgency
            new_dict_list.append(new_dict)
        elif dict_list[0].get('Type') in issue_type_set:
            status=[]
            urgency=[]
            conf = []
            segs = []

            for dic in dict_list:
                status.append(dic.get('Status'))
                urgency.append(dic.get('Urgent'))
                conf.append(dic.get('Confidence'))
                segs.append(dic.get('Justification_ID'))
            status = majority_ele_in_list(status)
            urgency = majority_ele_in_list(urgency)
            conf = max(conf)
            seg_id = best_seg_id(segs,2)
            new_dict={}
            new_dict['DocumentID'] = doc_id
            new_dict['Type'] = SF_type
            new_dict['Place_KB_ID'] =  kb_id
            new_dict['Status'] = status
            new_dict['Confidence'] = conf
            new_dict['Justification_ID'] = seg_id
            new_dict['Urgent'] = urgency
            new_dict_list.append(new_dict)
        else:
            print 'wring detected SF type:', dict_list[0].get('Type')
            exit(0)
    return       new_dict_list, len(ent_set)


def majority_ele_in_list(lis):
    c = Counter(lis)
    return c.most_common()[0][0]

def load_EDL2018_output(filename):
    '''
    Penn    IL9_NW_020591_20151021_I0040QK2J-7       [Kameruni]|Kameruni    IL9_NW_020591_20151021_I0040QK2J:411-418        2233387 GPE     NAM     1.0
    '''
    readfile = codecs.open(filename, 'r', 'utf-8')
    # readfile = codecs.open('/save/wenpeng/datasets/LORELEI/finalsubmission.tab', 'r', 'utf-8')
    docid2poslist = defaultdict(list)
    for line in readfile:
        parts = line.strip().split('\t')
        type = parts[5]
        kb_id = parts[4]
        if (type == 'GPE' or type == 'LOC') and kb_id.find('NULL')<0:
            comma_pos = parts[3].find(':')
            doc_id = parts[3][:comma_pos]
            pos_pair = parts[3][comma_pos+1:]
            docid2poslist[doc_id].append(pos_pair+'-'+kb_id)
    readfile.close()
    print 'load EDL2018 over, size:', len(docid2poslist)
    return docid2poslist
