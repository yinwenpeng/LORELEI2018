#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import json
from pprint import pprint
import nltk
from collections import defaultdict
import os
import xml.etree.ElementTree as ET
import string


def load_il6():
    '''
    monolingual_text: doc_id: sent_list:['...','...'], boundary_list:[(1,12),(14,23)...]
    ground truth:
        issue: doc_id:[{'entity_id': place_id, 'frame_type': issue, 'issue_type':crimeviolence},{'entity_id':...}]
        mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
        needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
    '''

    #first load il5 sent list
    folder = '/save/wenpeng/datasets/LORELEI/il6_setE_monolingual_text/ltf/'
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
    folder = '/save/wenpeng/datasets/LORELEI/il6_unseq/setE/data/annotation/situation_frame/issues/'
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
    folder = '/save/wenpeng/datasets/LORELEI/il6_unseq/setE/data/annotation/situation_frame/mentions/'
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
    folder = '/save/wenpeng/datasets/LORELEI/il6_unseq/setE/data/annotation/situation_frame/needs/'
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
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/il6_labeled_as_training_seg_level.txt', 'w', 'utf-8')
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
                    # labelstrlist.append(issue_type)
                    # idd = type2label_id.get(issue_type)
                    # if idd is None:
                    #     print 'issue_type is None:', issue_type
                    #     exit(0)
                    # iddlist.append(idd)

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
                    # labelstrlist.append(need_type)
                    # idd = type2label_id.get(need_type)
                    # if idd is None:
                    #     print 'need_type is None:', need_type
                    #     exit(0)
                    # iddlist.append(idd)


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

    folder = '/save/wenpeng/datasets/LORELEI/il6_unseq/setE/data/annotation/situation_frame/'
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


def denoise(text):
    tokens = text.split()
    new_text = []
    for token in tokens:
        if token.find('https://') < 0 and token.find('http://') < 0:
            new_text.append(token)
    # tweet.translate(string.maketrans("",""), string.punctuation)
    new_text = ' '.join(new_text)
    return new_text
def load_text_given_docvocab(doc_set):
    folder = '/save/wenpeng/datasets/LORELEI/il6_setE_monolingual_text/ltf/'
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

            sent = denoise(sent)
            if len(sent.strip())>0:
                docid2text[doc_id] = sent
            # f.close()
    print 'load load_text_given_docvocab over, size: ', len(docid2text)
    return docid2text

def store_il6_as_trainingformat(doc_2_labelstrlist,doc_2_text ):
    type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2, 'water':7, 'shelter':5,
    'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/il6_labeled_as_training.txt', 'w', 'utf-8')
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
    print 'store labeled il6 over'


def reformat_il6_dictionary():
    readfile = codecs.open('/shared/corpora/corporaWeb/lorelei/evaluation-20170804/LDC2017E29_LORELEI_IL6_Incident_Language_Pack_for_Year_2_Eval_V1.1/set0/docs/categoryI_dictionary/orm-eng.lorelei.txt', 'r', 'utf-8')
    writefile = codecs.open('/shared/corpora/corporaWeb/lorelei/evaluation-upenn-20180402/il6/source/set0/docs/categoryI_dictionary/IL6_dictionary.txt', 'w', 'utf-8')
    for line in readfile:
        parts=line.strip().split('\t')
        eng_token = parts[5]
        il_token = parts[0]
        writefile.write(eng_token+'\t'+il_token+'\n')
    print 'il6 dic reformat over'
    readfile.close()
    writefile.close()


if __name__ == '__main__':
    # doc_2_labelstrlist = load_docID_2_labelStr()
    # doc_2_text = load_text_given_docvocab(set(doc_2_labelstrlist.keys()))
    # store_il6_as_trainingformat(doc_2_labelstrlist, doc_2_text)

    # docid2text, docid2issue, docid2mention, docid2need = load_il6()
    # generate_entity_focused_trainingset(docid2text, docid2issue, docid2mention, docid2need)

    reformat_il6_dictionary()
