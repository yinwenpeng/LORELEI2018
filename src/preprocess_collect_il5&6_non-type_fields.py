import codecs
from preprocess_il5 import load_il5
from preprocess_il6 import load_il6
import os


import json

import nltk
from collections import defaultdict

import xml.etree.ElementTree as ET

'''
collect (two versions of) translated il5 and il6 for 8 needs+3issues, and all other fields
(needs&issue, status, relief, urgency, text)
'''

def load_il5_translated():
    read_mtfiles = codecs.open('/save/wenpeng/datasets/LORELEI/il5_unseq/setE/docs/annotated_filelist_MT.txt','r','utf-8')
    translated_docID_set=set()
    for line in read_mtfiles:
        translated_docID_set.add(line.strip())
    read_mtfiles.close()
    folder = '/save/wenpeng/datasets/LORELEI/il5_unseq/setE/data/translation/eng/ltf/'
    docid2text={}
    re=0

    files= set(os.listdir(folder))
    print 'folder file sizes: ', len(files)
    for trans_doc_id in translated_docID_set:
        v_A_name = trans_doc_id+'.eng_A.ltf.xml'
        v_B_name = trans_doc_id+'.eng_B.ltf.xml'
        doc_instance={}
        doc_instance['doc_id'] = trans_doc_id
        if v_A_name in files:
            sent_list_version1 = []
            f = ET.parse(folder+'/'+v_A_name)
            root = f.getroot()
            for seg in root.iter('SEG'):
                sent_i = seg.find('ORIGINAL_TEXT').text
                sent_list_version1.append(sent_i)

            doc_instance['sent_list_version1'] = sent_list_version1
        if v_B_name in files:
            sent_list_version2 = []
            f = ET.parse(folder+'/'+v_B_name)
            root = f.getroot()
            for seg in root.iter('SEG'):
                sent_i = seg.find('ORIGINAL_TEXT').text
                sent_list_version2.append(sent_i)

            doc_instance['sent_list_version2'] = sent_list_version2
        docid2text[trans_doc_id] = doc_instance
        # f.close()
    print 'load translated il5 over, size: ', len(docid2text)
    return docid2text

def load_il6_translated():
    read_mtfiles = codecs.open('/save/wenpeng/datasets/LORELEI/il6_unseq/setE/docs/annotated_filelist_MT.txt','r','utf-8')
    translated_docID_set=set()
    for line in read_mtfiles:
        translated_docID_set.add(line.strip())
    read_mtfiles.close()
    folder = '/save/wenpeng/datasets/LORELEI/il6_unseq/setE/data/translation/eng/ltf/'
    docid2text={}
    re=0

    files= set(os.listdir(folder))
    print 'folder file sizes: ', len(files)
    for trans_doc_id in translated_docID_set:
        v_A_name = trans_doc_id+'.eng_A.ltf.xml'
        v_B_name = trans_doc_id+'.eng_B.ltf.xml'
        doc_instance={}
        doc_instance['doc_id'] = trans_doc_id
        if v_A_name in files:
            sent_list_version1 = []
            f = ET.parse(folder+'/'+v_A_name)
            root = f.getroot()
            for seg in root.iter('SEG'):
                sent_i = seg.find('ORIGINAL_TEXT').text
                sent_list_version1.append(sent_i)

            doc_instance['sent_list_version1'] = sent_list_version1
        if v_B_name in files:
            sent_list_version2 = []
            f = ET.parse(folder+'/'+v_B_name)
            root = f.getroot()
            for seg in root.iter('SEG'):
                sent_i = seg.find('ORIGINAL_TEXT').text
                sent_list_version2.append(sent_i)

            doc_instance['sent_list_version2'] = sent_list_version2
        docid2text[trans_doc_id] = doc_instance
        # f.close()
    print 'load translated il6 over, size: ', len(docid2text)
    return docid2text

def entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id, window):

    mention_entity_instance_list = docid2mention.get(doc_id)
    for instance_dict in mention_entity_instance_list:
        if instance_dict.get('entity_id') == entity_id:
            start_char = instance_dict.get('start_char')
            end_char = instance_dict.get('end_char')
            # print start_char, end_char, boundary_list
            sent_size = len(boundary_list)
            for i in range(sent_size):
                tuplee = boundary_list[i]
                # print tuplee
                # print start_char, tuplee[1]
                # print tuplee[1] < start_char #,  start_char < tuplee[1]
                # print tuplee[1]/2.0
                # print start_char/2.0
                # exit(0)
                # print start_char, end_char, tuplee[0], tuplee[1], start_char >= tuplee[0], start_char <= 139, 34 <= tuplee[1], 34 <=139
                if start_char >= tuplee[0] and end_char <= tuplee[1]:
                    raw_ids = range(i-window, i+window+1)#[i-1,i,i+1]
                    final_ids = []
                    for iddd in raw_ids:
                        if iddd>=0 and iddd < sent_size:
                            final_ids.append(iddd)
                    return '-'.join(map(str, final_ids))
    print 'failed to find a sentence for the location:', entity_id
    exit(0)

def denoise(text):
    http_pos = text.find('https://')
    return text[:http_pos].strip()

def generate_entity_focused_trainingset_il5(docid2text, docid2trans,docid2issue, docid2mention, docid2need, window):
    '''
    monolingual_text: doc_id: sent_list:['...','...'], boundary_list:[(1,12),(14,23)...]
    ground truth:
        issue: doc_id:[{'entity_id': place_id, 'frame_type': issue, 'issue_type':crimeviolence},{'entity_id':...}]
        mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
        needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
    '''
    type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2, 'water':7, 'shelter':5,
    'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}
    # other_field2index = {'current':0,'not_current':1, 'sufficient':0,'insufficient':1,'True':0,'False':1}
    other_field2index = {'current':0,'not_current':1,'future':1,'past':2, 'sufficient':0,'insufficient':1,'True':0,'False':1}
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/il5_translated_seg_level_as_training_all_fields_w'+str(window)+'.txt', 'w', 'utf-8')
    write_size = 0
    doc_union_issue_and_needs = set(docid2issue.keys())| set(docid2need.keys())
    for doc_id, doc_instance in docid2text.iteritems():
        trans_instance = docid2trans.get(doc_id)
        if trans_instance is not None:
            sent_list = doc_instance.get('sent_list')
            trans_v1_sent_list = trans_instance.get('sent_list_version1')
            trans_v2_sent_list = trans_instance.get('sent_list_version2')
            boundary_list = doc_instance.get('boundary_list')

            if doc_id  in doc_union_issue_and_needs: #this doc has SF type labels
                # iddlist=[]
                # labelstrlist = []
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
                            sent_id = entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id, window)

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
                            sent_id = entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id, window)
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

                    idlist = sent_ids.split('-')
                    sent1 = ''
                    if trans_v1_sent_list is not None:
                        for id in idlist:
                            sent1+=' '+trans_v1_sent_list[int(id)]
                    sent2 = ''
                    if trans_v2_sent_list is not None:
                        for id in idlist:
                            sent2+=' '+trans_v2_sent_list[int(id)]

                    iddlist=[]
                    labelstrlist_delete_duplicate = list(set(labelstrlist))
                    for  labelstr in labelstrlist_delete_duplicate:
                        idd = type2label_id.get(labelstr)
                        if idd is None:
                            print 'labelstr is None:', labelstr
                            exit(0)
                        iddlist.append(idd)

                    if len(sent1) > 0:
                        writefile.write(' '.join(map(str, iddlist))+'\t'+' '.join(labelstrlist_delete_duplicate)+'\t'+denoise(sent1.strip())+'\t'+' '.join(map(str,other_fields))+'\n')
                    if len(sent2) > 0:
                        writefile.write(' '.join(map(str, iddlist))+'\t'+' '.join(labelstrlist_delete_duplicate)+'\t'+denoise(sent2.strip())+'\t'+' '.join(map(str,other_fields))+'\n')
                    write_size+=1
    writefile.close()
    print 'write_size:', write_size

def generate_entity_focused_trainingset_il6(docid2text, docid2trans,docid2issue, docid2mention, docid2need, window):
    '''
    monolingual_text: doc_id: sent_list:['...','...'], boundary_list:[(1,12),(14,23)...]
    ground truth:
        issue: doc_id:[{'entity_id': place_id, 'frame_type': issue, 'issue_type':crimeviolence},{'entity_id':...}]
        mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
        needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
    '''
    type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2, 'water':7, 'shelter':5,
    'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}
    other_field2index = {'current':0,'not_current':1,'future':1,'past':2, 'sufficient':0,'insufficient':1,'True':0,'False':1}
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/il6_translated_seg_level_as_training_all_fields_w'+str(window)+'.txt', 'w', 'utf-8')
    write_size = 0
    doc_union_issue_and_needs = set(docid2issue.keys())| set(docid2need.keys())
    for doc_id, doc_instance in docid2text.iteritems():
        trans_instance = docid2trans.get(doc_id)
        if trans_instance is not None:
            sent_list = doc_instance.get('sent_list')
            trans_v1_sent_list = trans_instance.get('sent_list_version1')
            trans_v2_sent_list = trans_instance.get('sent_list_version2')
            boundary_list = doc_instance.get('boundary_list')

            if doc_id  in doc_union_issue_and_needs: #this doc has SF type labels
                # iddlist=[]
                # labelstrlist = []
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
                            sent_id = entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id, window)

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
                            sent_id = entity_id_2_sentID(sent_list, boundary_list, docid2mention, doc_id, entity_id, window)
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

                    idlist = sent_ids.split('-')
                    sent1 = ''
                    if trans_v1_sent_list is not None:
                        for id in idlist:
                            sent1+=' '+trans_v1_sent_list[int(id)]
                    sent2 = ''
                    if trans_v2_sent_list is not None:
                        for id in idlist:
                            sent2+=' '+trans_v2_sent_list[int(id)]

                    iddlist=[]
                    labelstrlist_delete_duplicate = list(set(labelstrlist))
                    for  labelstr in labelstrlist_delete_duplicate:
                        idd = type2label_id.get(labelstr)
                        if idd is None:
                            print 'labelstr is None:', labelstr
                            exit(0)
                        iddlist.append(idd)

                    if len(sent1) > 0:
                        writefile.write(' '.join(map(str, iddlist))+'\t'+' '.join(labelstrlist_delete_duplicate)+'\t'+denoise(sent1.strip())+'\t'+' '.join(map(str,other_fields))+'\n')
                    if len(sent2) > 0:
                        writefile.write(' '.join(map(str, iddlist))+'\t'+' '.join(labelstrlist_delete_duplicate)+'\t'+denoise(sent2.strip())+'\t'+' '.join(map(str,other_fields))+'\n')
                    write_size+=1
    writefile.close()
    print 'write_size:', write_size

if __name__ == '__main__':
    # docid2text, docid2issue, docid2mention, docid2need = load_il5()
    # docid2trans = load_il5_translated()
    # generate_entity_focused_trainingset_il5(docid2text, docid2trans,docid2issue, docid2mention, docid2need,1)

    docid2text, docid2issue, docid2mention, docid2need = load_il6()
    docid2trans = load_il6_translated()
    generate_entity_focused_trainingset_il6(docid2text, docid2trans,docid2issue, docid2mention, docid2need, 1)
