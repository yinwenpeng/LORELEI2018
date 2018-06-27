import codecs
import json
from pprint import pprint
import nltk
from collections import defaultdict
import os
import xml.etree.ElementTree as ET


def valid_seg(doc_id, seg_id,doc2remain_seglist):

    if doc_id in doc2remain_seglist:
        seg_list = doc2remain_seglist.get(doc_id)
        for gold_seg_id in seg_list:

            if seg_id == gold_seg_id:
                return True
                break
    return False
def Thai_into_test_filteredby_NER(doc2remain_seglist):
    folder = '/save/wenpeng/datasets/LORELEI/Thai/setE/'
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/Thai/thai-setE-as-test-input_ner_filtered.txt', 'w', 'utf-8')
    vocab=set()
    re=0

    files= os.listdir(folder)
    print 'folder file sizes: ', len(files)
    co=0
    for fil in files:
        sent_list = []
        # print folder+'/'+fil
        f = ET.parse(folder+fil)
        # f = codecs.open(folder+'/'+fil, 'r', 'utf-8')

        root = f.getroot()
        for doc in root.iter('DOC'):
            doc_id  = doc.attrib.get('id')
        for seg in root.iter('SEG'):
            sent_wordlist = []
            seg_id = seg.attrib.get('id')
            for word in seg.iter('TOKEN'):
                word_str = word.text
                sent_wordlist.append(word_str)
            if valid_seg(doc_id, seg_id,doc2remain_seglist):
                writefile.write(doc_id+'\t'+seg_id+'\t'+' '.join(sent_wordlist)+'\n')
                co+=1
                if co % 1000 == 0:
                    print 'co...', co
    writefile.close()
    print 'Thai_into_test over'

def Thai_into_test():
    folder = '/save/wenpeng/datasets/LORELEI/Thai/setE/'
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/Thai/thai-setE-as-test-input.txt', 'w', 'utf-8')
    vocab=set()
    re=0

    files= os.listdir(folder)
    print 'folder file sizes: ', len(files)
    co=0
    for fil in files:
        sent_list = []
        # print folder+'/'+fil
        f = ET.parse(folder+fil)
        # f = codecs.open(folder+'/'+fil, 'r', 'utf-8')

        root = f.getroot()
        for doc in root.iter('DOC'):
            doc_id  = doc.attrib.get('id')
        for seg in root.iter('SEG'):
            sent_wordlist = []
            seg_id = seg.attrib.get('id')
            for word in seg.iter('TOKEN'):
                word_str = word.text
                sent_wordlist.append(word_str)
            writefile.write(doc_id+'\t'+seg_id+'\t'+' '.join(sent_wordlist)+'\n')
            co+=1
            if co % 1000 == 0:
                print 'co...', co
    writefile.close()
    print 'Thai_into_test over'

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

def denoise(text):
    http_pos = text.find('https')
    return text[:http_pos].strip()

def load_Thai_annotations():
    '''
    monolingual_text: doc_id: sent_list:['...','...'], boundary_list:[(1,12),(14,23)...]
    ground truth:
        issue: doc_id:[{'entity_id': place_id, 'frame_type': issue, 'issue_type':crimeviolence},{'entity_id':...}]
        mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
        needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
    '''

    #first load il5 sent list
    folder = '/save/wenpeng/datasets/LORELEI/Thai/setE/'
    docid2text={}
    re=0

    files= os.listdir(folder)
    print 'folder file sizes: ', len(files)
    for fil in files:

        boundary_list = []
        sent_list = []
        seg_idlist = []
        f = ET.parse(folder+'/'+fil)
        # f = codecs.open(folder+'/'+fil, 'r', 'utf-8')

        root = f.getroot()
        for doc in root.iter('DOC'):
            doc_id  = doc.attrib.get('id')
        for seg in root.iter('SEG'):
            seg_id = seg.attrib.get('id')
            start = int(seg.attrib.get('start_char'))
            end = int(seg.attrib.get('end_char'))
            sent_wordlist = []
            for word in seg.iter('TOKEN'):
                word_str = word.text
                sent_wordlist.append(word_str)
            sent_i = ' '.join(sent_wordlist)
            # sent+=' '+sent_i
            sent_list.append(sent_i)
            boundary_list.append((start,end))
            seg_idlist.append(seg_id)
        doc_instance={}
        doc_instance['doc_id'] = doc_id
        doc_instance['sent_list'] = sent_list
        doc_instance['boundary_list'] = boundary_list
        doc_instance['seg_idlist'] = seg_idlist

        docid2text[doc_id] = doc_instance
        # f.close()
    print 'load load_text_given_docvocab over, size: ', len(docid2text)

    '''
    load issues
    '''
    docid2issue = {}
    folder = '/save/wenpeng/datasets/LORELEI/Thai/annotation/situation_frame/issues/'
    files= os.listdir(folder)
    print 'issues file sizes: ', len(files)
    for fil in files:
        if not os.path.isdir(fil):
            f = codecs.open(folder+'/'+fil, 'r', 'utf-8')
            line_co = 0

            issue_list = []
            for line in f:
                if line_co == 0:
                    line_co+=1
                    continue
                else:

                    issue_instance = {}
                    parts = line.strip().split('\t')
                    doc_id = parts[1]
                    frame_type = parts[3]
                    issue_type = parts[4]
                    place_id = parts[5]
                    issue_status = parts[7]

                    # issue_instance['doc_id'] = doc_id
                    issue_instance['frame_type'] = frame_type
                    issue_instance['issue_type'] = issue_type
                    issue_instance['entity_id'] = place_id
                    issue_instance['issue_status'] = issue_status
                    issue_list.append(issue_instance)
                    line_co+=1
            issue_list_remove_duplicate = {frozenset(item.items()):item for item in issue_list}.values()
            docid2issue[doc_id] = issue_list_remove_duplicate
            f.close()
    '''
    load mentions
    mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
    '''
    folder = '/save/wenpeng/datasets/LORELEI/Thai/annotation/situation_frame/mentions/'
    files= os.listdir(folder)
    print 'mentions file sizes: ', len(files)
    docid2mention = {}
    for fil in files:
        if not os.path.isdir(fil):
            f = codecs.open(folder+'/'+fil, 'r', 'utf-8')
            line_co = 0
            mention_list = []
            for line in f:
                if line_co == 0:
                    line_co+=1
                    continue
                else:

                    mention_instance = {}
                    parts = line.strip().split('\t')
                    doc_id = parts[0]
                    entity_id = parts[1]
                    entity_type = parts[3]
                    start_char = parts[5]
                    end_char = parts[6]


                    # mention_instance['doc_id'] = doc_id
                    mention_instance['entity_id'] = entity_id
                    mention_instance['entity_type'] = entity_type
                    mention_instance['start_char'] = int(start_char)
                    mention_instance['end_char'] = int(end_char)

                    mention_list.append(mention_instance)
                    line_co+=1
            mention_list_remove_duplicate = {frozenset(item.items()):item for item in mention_list}.values()
            docid2mention[doc_id] = mention_list_remove_duplicate
            # print 'docid2mention[doc_id]:',doc_id,docid2mention[doc_id]
            # exit(0)
            f.close()

    '''
    load needs
    needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
    user_id doc_id  frame_id        frame_type      need_type       place_id        proxy_status    need_status     urgency_status  resolution_status       reported_by     resolved_by     description
    '''
    docid2need = {}
    folder = '/save/wenpeng/datasets/LORELEI/Thai/annotation/situation_frame/needs/'
    files= os.listdir(folder)
    print 'needs file sizes: ', len(files)
    for fil in files:
        if not os.path.isdir(fil):
            f = codecs.open(folder+'/'+fil, 'r', 'utf-8')
            line_co = 0

            need_list = []
            for line in f:
                if line_co == 0:
                    line_co+=1
                    continue
                else:

                    need_instance = {}
                    parts = line.strip().split('\t')
                    doc_id = parts[1]
                    frame_type = parts[3]
                    need_type = parts[4]
                    place_id = parts[5]
                    need_status = parts[7]
                    urgency_status = parts[8]
                    resolution_status = parts[9]

                    # issue_instance['doc_id'] = doc_id
                    need_instance['frame_type'] = frame_type
                    need_instance['need_type'] = need_type
                    need_instance['entity_id'] = place_id
                    need_instance['need_status'] = need_status
                    need_instance['urgency_status'] = urgency_status
                    need_instance['resolution_status'] = resolution_status


                    need_list.append(need_instance)
                    line_co+=1
            need_list_remove_duplicate = {frozenset(item.items()):item for item in need_list}.values()
            docid2need[doc_id] = need_list_remove_duplicate
            # print doc_id, docid2need[doc_id]
            # exit(0)
            f.close()
    return docid2text, docid2issue, docid2mention, docid2need


def entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id):

    mention_entity_instance_list = docid2mention.get(doc_id)
    for instance_dict in mention_entity_instance_list:
        if instance_dict.get('entity_id') == entity_id:
            start_char = instance_dict.get('start_char')
            end_char = instance_dict.get('end_char')
            # print start_char, end_char, boundary_list
            for i in range(len(boundary_list)):
                tuplee = boundary_list[i]
                # print tuplee
                # print start_char, tuplee[1]
                # print tuplee[1] < start_char #,  start_char < tuplee[1]
                # print tuplee[1]/2.0
                # print start_char/2.0
                # exit(0)
                # print start_char, end_char, tuplee[0], tuplee[1], start_char >= tuplee[0], start_char <= 139, 34 <= tuplee[1], 34 <=139
                if start_char >= tuplee[0] and end_char <= tuplee[1]:
                    return '-'.join(map(str, [i]))
    print 'failed to find a sentence for the location:', entity_id
    exit(0)


def generate_entity_focused_trainingset(docid2text, docid2issue, docid2mention, docid2need):
    '''
    monolingual_text: doc_id: sent_list:['...','...'], boundary_list:[(1,12),(14,23)...]
    ground truth:
        issue: doc_id:[{'entity_id': place_id, 'frame_type': issue, 'issue_type':crimeviolence},{'entity_id':...}]
        mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
        needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
    '''
    type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2, 'water':7, 'shelter':5,
    'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}
    other_field2index = {'current':0,'not_current':1,'future':1,'past':2,  'sufficient':0,'insufficient':1,'True':0,'False':1}
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/Thai/thai_test_with_full_labels_seg_level.txt', 'w', 'utf-8')
    write_size = 0
    doc_union_issue_and_needs = set(docid2issue.keys())| set(docid2need.keys())
    for doc_id, doc_instance in docid2text.iteritems():

        sent_list = doc_instance.get('sent_list')
        # trans_v1_sent_list = trans_instance.get('sent_list_version1')
        # trans_v2_sent_list = trans_instance.get('sent_list_version2')
        boundary_list = doc_instance.get('boundary_list')
        if doc_id  in doc_union_issue_and_needs: #this doc has SF type labels
            sentID_2_labelstrlist=defaultdict(list)
            other_fields=[3]*4 #need_status, issue_status, need_relief, need_urgency, defalse "2" denotes no label
            issue_list = docid2issue.get(doc_id)
            if issue_list is not None:
                for i in range(len(issue_list)):
                    issue_dict_instance = issue_list[i]
                    entity_id = issue_dict_instance.get('entity_id')
                    if entity_id == 'none':
                        sent_id = '-'.join(map(str,range(len(sent_list))))
                    else:
                        sent_id = entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id)

                    issue_type = issue_dict_instance.get('issue_type')
                    sentID_2_labelstrlist[sent_id].append(issue_type)
                    issue_status = issue_dict_instance.get('issue_status')
                    if other_field2index.get(issue_status) is None:
                        print 'issue_status:',issue_status
                    other_fields[1] = other_field2index.get(issue_status,3)


            need_list = docid2need.get(doc_id)
            if need_list is not None:
                for i in range(len(need_list)):
                    need_dict_instance = need_list[i]
                    entity_id = need_dict_instance.get('entity_id')
                    if entity_id == 'none':
                        sent_id = '-'.join(map(str,range(len(sent_list))))
                    else:
                        sent_id = entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id)
                    need_type = need_dict_instance.get('need_type')
                    sentID_2_labelstrlist[sent_id].append(need_type)

                    need_status = need_dict_instance.get('need_status')
                    need_relief = need_dict_instance.get('resolution_status')
                    need_urgency = need_dict_instance.get('urgency_status')
                    if other_field2index.get(need_urgency) is None:
                        print 'need_urgency:',need_urgency
                    if other_field2index.get(need_status) is None:
                        print 'need_status:',need_status
                    if other_field2index.get(need_relief) is None:
                        print 'need_relief:',need_relief
                    other_fields[0] = other_field2index.get(need_status,3)
                    other_fields[2] = other_field2index.get(need_relief,3)
                    other_fields[3] = other_field2index.get(need_urgency,3)


            for sent_ids, labelstrlist in sentID_2_labelstrlist.iteritems():
                sent = ''
                idlist = sent_ids.split('-')
                for id in idlist:
                    sent+=' '+sent_list[int(id)]

                iddlist=[]
                labelstrlist_delete_duplicate = list(set(labelstrlist))
                for  labelstr in labelstrlist_delete_duplicate:
                    idd = type2label_id.get(labelstr)
                    if idd is None:
                        print 'labelstr is None:', labelstr
                        exit(0)
                    iddlist.append(idd)
                writefile.write(' '.join(map(str, iddlist))+'\t'+' '.join(labelstrlist_delete_duplicate)+'\t'+denoise(sent.strip())+'\t'+' '.join(map(str,other_fields))+'\n')
                write_size+=1
    writefile.close()
    print 'write_size:', write_size

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

def valid_SF(new_dict,doc2remain_seglist):
    # print 'new_dict:', new_dict

    doc_id = new_dict.get('DocumentID')
    # print 'doc2remain_seglist:', doc2remain_seglist.get(doc_id)
    # exit(0)
    seg_id = new_dict.get('Justification')
    if doc_id in doc2remain_seglist:
        seg_list = doc2remain_seglist.get(doc_id)
        seg_set = set()
        for gold_seg_id in seg_list:
            # gold_seg_id = seg.get('id')
            if seg_id == gold_seg_id:
                return True
                break
    return False

def generate_official_output(pred_types, pred_confs, pred_others, min_mean_frame):
    #pred_others (#instance, 4, 3)
    thai_root = '/save/wenpeng/datasets/LORELEI/Thai/'
    instance_size = len(pred_types)
    type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2,
    'water':7, 'shelter':5, 'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}

    id2type = {y:x for x,y in type2label_id.iteritems()}
    readfile = codecs.open(thai_root+'thai-setE-as-test-input_ner_filtered.txt', 'r', 'utf-8')

    output_dict_list = []
    lines=[]
    for line in readfile:
        lines.append(line.strip())
    readfile.close()
    assert instance_size == len(pred_others)
    assert instance_size == len(pred_confs)
    assert instance_size == len(lines)
    print 'seg size to pred: ', instance_size, 'full file size:', len(lines)
    # assert instance_size == len(lines)


    # pred_needs = pred_types[:,:8]
    # pred_issues = np.concatenate([pred_types[:,8:9], pred_types[:, 10:]),axis=1)  #(all, 3)

    #needs
    for i in range(instance_size):
        # print 'line...', i
        pred_vec = list(pred_types[i])
        text_parts = lines[i].split('\t')
        doc_id = text_parts[0]
        seg_id = text_parts[1]
        for x, y in enumerate(pred_vec):
            if y == 1:
                if x < 8: # is a need type
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
                    output_dict_list.append(new_dict)
                elif x ==8 or x > 9: # is issue
                    new_dict={}
                    new_dict['DocumentID'] = doc_id
                    hit_issue_type = id2type.get(x)
                    new_dict['Type'] = hit_issue_type
                    new_dict['Place_KB_ID'] = 'TBD'
                    status, urgency = get_issue_other_fields(pred_others[i])
                    new_dict['Status'] = status
                    new_dict['Confidence'] = float(pred_confs[i][x])
                    new_dict['Justification_ID'] = seg_id
                    new_dict['Urgent'] = urgency
                    output_dict_list.append(new_dict)

    frame_size = len(output_dict_list)
    mean_frame = frame_size*1.0/instance_size
    if mean_frame < min_mean_frame:
        writefile = codecs.open(thai_root+'system_output_forfun.json' ,'w', 'utf-8')
        json.dump(output_dict_list, writefile)
        writefile.close()
        print '............new loweast written over'

    print 'official output succeed...Frame size:', frame_size, 'average:', mean_frame
    return mean_frame


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




if __name__ == '__main__':
    # Thai_into_test()
    '''
    the following method to create annotated test data does not make sense
    '''
    docid2text, docid2issue, docid2mention, docid2need = load_Thai_annotations()
    generate_entity_focused_trainingset(docid2text, docid2issue, docid2mention, docid2need)

    # doc2entitylist = preprocess_NER_results()
    # doc2seglist = Thai_rich_info()
    # filter_ner_results(doc2entitylist, doc2seglist)

    # doc2remain_seglist = filter_ner_results()
    # Thai_into_test_filteredby_NER(doc2remain_seglist)
