import codecs
import json
from pprint import pprint
import nltk
from collections import defaultdict,Counter
import os
import xml.etree.ElementTree as ET
from preprocess_common import IL_into_test_filteredby_NER,SF_annotated_docSet

def il5_into_text_corpus_for_word2vec_training():
    folder = '/save/wenpeng/datasets/LORELEI/il5_setE_monolingual_text/ltf/'
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/multi-lingual-emb/il5_monolingual_text.txt', 'w', 'utf-8')
    vocab=set()
    re=0

    files= os.listdir(folder)
    print 'folder file sizes: ', len(files)
    for fil in files:
        sent_list = []
        f = ET.parse(folder+'/'+fil)
        # f = codecs.open(folder+'/'+fil, 'r', 'utf-8')

        root = f.getroot()
        for seg in root.iter('SEG'):
            sent_i = seg.find('ORIGINAL_TEXT').text
            vocab=vocab|set(sent_i.split())
            writefile.write(sent_i.strip()+'\n')
    writefile.close()
    print 'load il5 text over, size: ', len(vocab)

def load_il5():
    '''
    monolingual_text: doc_id: sent_list:['...','...'], boundary_list:[(1,12),(14,23)...]
    ground truth:
        issue: doc_id:[{'entity_id': place_id, 'frame_type': issue, 'issue_type':crimeviolence},{'entity_id':...}]
        mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
        needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
    '''

    #first load il5 sent list
    folder = '/save/wenpeng/datasets/LORELEI/il5_setE_monolingual_text/ltf/'
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
            sent_i = seg.find('ORIGINAL_TEXT').text
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
    folder = '/save/wenpeng/datasets/LORELEI/il5_unseq/setE/data/annotation/situation_frame/issues/'
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
    folder = '/save/wenpeng/datasets/LORELEI/il5_unseq/setE/data/annotation/situation_frame/mentions/'
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
    folder = '/save/wenpeng/datasets/LORELEI/il5_unseq/setE/data/annotation/situation_frame/needs/'
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
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/il5_labeled_as_training_seg_level.txt', 'w', 'utf-8')
    write_size = 0
    for doc_id, doc_instance in docid2text.iteritems():
        sent_list = doc_instance.get('sent_list')
        boundary_list = doc_instance.get('boundary_list')
        doc_uninion_issue_and_mentions = set(docid2issue.keys())| set(docid2need.keys())
        if doc_id  in doc_uninion_issue_and_mentions: #this doc has SF type labels
            # iddlist=[]
            # labelstrlist = []
            sentID_2_labelstrlist=defaultdict(list)
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

                writefile.write(' '.join(map(str, iddlist))+'\t'+' '.join(labelstrlist_delete_duplicate)+'\t'+sent.strip()+'\n')
                write_size+=1
    writefile.close()
    print 'write_size:', write_size






def load_docID_2_labelStr():

    folder = '/save/wenpeng/datasets/LORELEI/il5_unseq/setE/data/annotation/situation_frame/'
    subfoldernames = ['issues', 'needs']
    docid2labelstr=defaultdict(list)
    re=0
    for subfolder in subfoldernames:
        true_folder = folder+subfolder

        files= os.listdir(true_folder)
        print 'folder file sizes: ', len(files)
        for fil in files:
            if not os.path.isdir(fil):
                f = codecs.open(true_folder+'/'+fil, 'r', 'utf-8')
                line_co = 0
                for line in f:
                    if line_co == 0:
                        line_co+=1
                        continue
                    else:
                        parts = line.strip().split('\t')
                        doc_id = parts[1]
                        issue_str = parts[4]
                        if doc_id in docid2labelstr:
                            re+=1
                        docid2labelstr[doc_id].append(issue_str)
                        line_co+=1
                        break
                f.close()
    print 'load load_docID_2_labelStr over, size: ', len(docid2labelstr), 're size: ', re
    return docid2labelstr

def load_text_given_docvocab(doc_set):
    folder = '/save/wenpeng/datasets/LORELEI/il5_setE_monolingual_text/ltf/'
    docid2text={}
    re=0

    files= os.listdir(folder)
    print 'folder file sizes: ', len(files)
    for fil in files:

        doc_id = fil[:fil.find('.')]
        if doc_id in doc_set:
            sent = ''
            f = ET.parse(folder+'/'+fil)
            # f = codecs.open(folder+'/'+fil, 'r', 'utf-8')
            root = f.getroot()
            for seg in root.iter('SEG'):
                sent_i = seg.find('ORIGINAL_TEXT').text
                sent+=' '+sent_i
            docid2text[doc_id] = sent
            # f.close()
    print 'load load_text_given_docvocab over, size: ', len(docid2text)
    return docid2text

def store_il5_as_trainingformat(doc_2_labelstrlist,doc_2_text ):
    type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2, 'water':7, 'shelter':5,
    'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/il5_labeled_as_training.txt', 'w', 'utf-8')
    for doc in doc_2_text:
        labelstrlist = doc_2_labelstrlist.get(doc)
        if labelstrlist is not None:
            # print 'labelstr: ', labelstr
            iddlist=[]
            for  labelstr in labelstrlist:
                idd = type2label_id.get(labelstr)
                if idd is None:
                    print 'labelstr is None:', labelstr
                    exit(0)
                iddlist.append(idd)

            writefile.write(' '.join(map(str, iddlist))+'\t'+' '.join(labelstrlist)+'\t'+doc_2_text.get(doc)+'\n')
    writefile.close()
    print 'store labeled il5 over'


def train_monolingual_emb_il5():
    os.system("./word2vec -train /save/wenpeng/datasets/LORELEI/multi-lingual-emb/il5_monolingual_text.txt -output /save/wenpeng/datasets/LORELEI/multi-lingual-emb/il5_300d_word2vec.txt -cbow 0 -size 300 -window 5 -negative 5 -hs 0 -iter 5 -sample 1e-3 -threads 12 -binary 0 -min-count 1")


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
def generate_2017_official_output(lines, output_file_path, pred_types, pred_confs, pred_others, min_mean_frame):
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
        entity_pos_list = lines[i].split('\t')[3].split()
        pred_vec = list(pred_types[i])
        text_parts = lines[i].split('\t')
        doc_id = text_parts[0]
        seg_id = text_parts[1]
        for x, y in enumerate(pred_vec):
            if y == 1:
                if x < 8: # is a need type
                    for entity_pos in entity_pos_list:
                        ent_start = int(entity_pos.split('-')[0])
                        ent_end = int(entity_pos.split('-')[1])
                        ent_type = entity_pos.split('-')[2]
                        new_dict={}
                        new_dict['DocumentID'] = doc_id
                        hit_need_type = id2type.get(x)
                        new_dict['Type'] = hit_need_type
                        new_dict['PlaceMention'] =  {'EntityType':ent_type, 'Start':ent_start, 'End':ent_end}
                        status,  relief,urgency = get_need_other_fields(pred_others[i])
                        new_dict['Status'] = {'Need':status, 'Relief':relief,'Urgent':urgency}
                        new_dict['TypeConfidence'] = float(pred_confs[i][x])
                        if new_dict.get('TypeConfidence') > 0.4:
                            output_dict_list.append(new_dict)
                elif x ==8 or x > 9: # is issue
                    for entity_pos in entity_pos_list:
                        ent_start = int(entity_pos.split('-')[0])
                        ent_end = int(entity_pos.split('-')[1])
                        ent_type = entity_pos.split('-')[2]
                        new_dict={}
                        new_dict['DocumentID'] = doc_id
                        hit_issue_type = id2type.get(x)
                        new_dict['Type'] = hit_issue_type
                        new_dict['PlaceMention'] =  {'EntityType':ent_type, 'Start':ent_start, 'End':ent_end}
                        status, urgency = get_issue_other_fields(pred_others[i])
                        new_dict['Status'] = {'Issue':status}
                        new_dict['TypeConfidence'] = float(pred_confs[i][x])
                        if new_dict.get('TypeConfidence') > 0.4:
                            output_dict_list.append(new_dict)


    refine_output_dict_list, ent_size = de_duplicate(output_dict_list)
    frame_size = len(refine_output_dict_list)
    mean_frame = frame_size*1.0/ent_size
    # if mean_frame < min_mean_frame:
    writefile = codecs.open(output_file_path ,'w', 'utf-8')
    json.dump(refine_output_dict_list, writefile)
    writefile.close()
    print '............new loweast written over'

    print 'official output succeed...Frame size:', frame_size, 'average:', mean_frame, 'ent_size:',ent_size
    return mean_frame

def de_duplicate(output_dict_list):
    need_type_set = set([ 'med','search','food','infra','water','shelter','evac','utils'])
    issue_type_set = set(['regimechange','crimeviolence','terrorism'])
    new_dict_list=[]
    key2dict_list = defaultdict(list)
    ent_set = set()
    for dic in output_dict_list:
        doc_id = dic.get('DocumentID')
        type = dic.get('Type')
        ent_start = dic.get('PlaceMention').get('Start')
        ent_end = dic.get('PlaceMention').get('End')
        key = (doc_id, type, ent_start,ent_end)
        ent_set.add((doc_id, ent_start,ent_end))
        key2dict_list[key].append(dic)
    for key, dict_list in key2dict_list.iteritems():
        #compute status, confidence
        if dict_list[0].get('Type') in need_type_set:
            status=[]
            relief=[]
            urgency=[]
            conf = []

            for dic in dict_list:
                status.append(dic.get('Status').get('Need'))
                relief.append(dic.get('Status').get('Relief'))
                urgency.append(dic.get('Status').get('Urgent'))
                conf.append(dic.get('TypeConfidence'))
            status = majority_ele_in_list(status)
            relief = majority_ele_in_list(relief)
            urgency = majority_ele_in_list(urgency)
            conf = max(conf)
            new_dict={}
            new_dict['DocumentID'] = dict_list[0].get('DocumentID')
            new_dict['Type'] = dict_list[0].get('Type')
            new_dict['PlaceMention'] =  dict_list[0].get('PlaceMention')
            new_dict['Status'] = {'Need':status, 'Relief':relief,'Urgent':urgency}
            new_dict['TypeConfidence'] = conf
            new_dict_list.append(new_dict)
        elif dict_list[0].get('Type') in issue_type_set:
            status=[]
            conf = []

            for dic in dict_list:
                status.append(dic.get('Status').get('Issue'))
                conf.append(dic.get('TypeConfidence'))
            status = majority_ele_in_list(status)
            conf = max(conf)
            new_dict={}
            new_dict['DocumentID'] = dict_list[0].get('DocumentID')
            new_dict['Type'] = dict_list[0].get('Type')
            new_dict['PlaceMention'] =  dict_list[0].get('PlaceMention')
            new_dict['Status'] = {'Issue':status}
            new_dict['TypeConfidence'] = conf
            new_dict_list.append(new_dict)
        else:
            print 'wring detected SF type:', dict_list[0].get('Type')
            exit(0)
    return       new_dict_list, len(ent_set)



def generate_output_for_EDL(lines, output_file_path, pred_types, pred_confs, pred_others, min_mean_frame):
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
        entity_pos_list = lines[i].split('\t')[3].split()
        pred_vec = list(pred_types[i])
        text_parts = lines[i].split('\t')
        doc_id = text_parts[0]
        seg_id = text_parts[1]
        for x, y in enumerate(pred_vec):
            if y == 1:
                if x < 8: # is a need type
                    for entity_pos in entity_pos_list:
                        ent_start = int(entity_pos.split('-')[0])
                        ent_end = int(entity_pos.split('-')[1])
                        ent_type = entity_pos.split('-')[2]
                        new_dict={}
                        new_dict['DocumentID'] = doc_id
                        hit_need_type = id2type.get(x)
                        new_dict['Type'] = hit_need_type
                        new_dict['Mid-seg-id'] =  i
                        new_dict['TypeConfidence'] = float(pred_confs[i][x])
                        if new_dict.get('TypeConfidence') > 0.4:
                            output_dict_list.append(new_dict)
                elif x ==8 or x > 9: # is issue
                    for entity_pos in entity_pos_list:
                        ent_start = int(entity_pos.split('-')[0])
                        ent_end = int(entity_pos.split('-')[1])
                        ent_type = entity_pos.split('-')[2]
                        new_dict={}
                        new_dict['DocumentID'] = doc_id
                        hit_issue_type = id2type.get(x)
                        new_dict['Type'] = hit_issue_type
                        new_dict['Mid-seg-id'] =  i
                        new_dict['TypeConfidence'] = float(pred_confs[i][x])
                        if new_dict.get('TypeConfidence') > 0.4:
                            output_dict_list.append(new_dict)



    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/il5_system_output_for_EDL.json' ,'w', 'utf-8')
    json.dump(output_dict_list, writefile)
    writefile.close()
    print '............new loweast written over, size:', len(output_dict_list)

    # print 'official output succeed...Frame size:', frame_size, 'average:', mean_frame, 'ent_size:',ent_size
    return 0.0

def majority_ele_in_list(lis):
    c = Counter(lis)
    return c.most_common()[0][0]




def load_EDL2017_output():
    readfile = codecs.open('/save/wenpeng/datasets/LORELEI/2017_best_run10-final.tab', 'r', 'utf-8')
    # readfile = codecs.open('/save/wenpeng/datasets/LORELEI/finalsubmission.tab', 'r', 'utf-8')
    docid2poslist = defaultdict(list)
    for line in readfile:
        parts = line.strip().split('\t')
        type = parts[5]
        if type == 'GPE' or type == 'LOC':
            comma_pos = parts[3].find(':')
            doc_id = parts[3][:comma_pos]
            pos_pair = parts[3][comma_pos+1:]
            docid2poslist[doc_id].append(pos_pair+'-'+type)
    readfile.close()
    print 'load EDL2017 over, size:', len(docid2poslist)
    return docid2poslist


if __name__ == '__main__':
    # doc_2_labelstrlist = load_docID_2_labelStr()
    # doc_2_text = load_text_given_docvocab(set(doc_2_labelstrlist.keys()))
    # store_il5_as_trainingformat(doc_2_labelstrlist, doc_2_text)
    '''
    above not useful anymore
    '''
    # docid2text, docid2issue, docid2mention, docid2need = load_il5()
    # generate_entity_focused_trainingset(docid2text, docid2issue, docid2mention, docid2need)
    # il5_into_text_corpus_for_word2vec_training()
    # train_monolingual_emb_il5()

    '''
    need the annotated docs and EDL docs to filter uncovered docs
    '''
    docset = SF_annotated_docSet('/save/wenpeng/datasets/LORELEI/official_scorer/annotated_filelist_SF.tab')
    docid2entity_pos_list = load_EDL2017_output()
    IL_into_test_filteredby_NER('/save/wenpeng/datasets/LORELEI/il5_setE_monolingual_text/ltf/',docset, '/save/wenpeng/datasets/LORELEI/il5-setE-as-test-input_ner_filtered', docid2entity_pos_list, 2)
