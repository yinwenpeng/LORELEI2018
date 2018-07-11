from collections import defaultdict,Counter

import json
import codecs
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

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

def de_duplicate(output_dict_list, key_count):
    need_type_set = set([ 'med','search','food','infra','water','shelter','evac','utils'])
    issue_type_set = set(['regimechange','crimeviolence','terrorism'])
    new_dict_list=[]
    key2dict_list = defaultdict(list)
    for dic in output_dict_list:
        doc_id = dic.get('DocumentID')
        type = dic.get('Type')
        kb_id = dic.get('Place_KB_ID')
        key = (doc_id, type, kb_id)
        if key in key_count:
            key2dict_list[key].append(dic)
    if len(key2dict_list)>0:
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
                new_dict['Confidence'] = sigmoid(conf*key_count.get(key))
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
                new_dict['Confidence'] = sigmoid(conf*key_count.get(key))
                new_dict['Justification_ID'] = seg_id
                new_dict['Urgent'] = urgency
                new_dict_list.append(new_dict)
            else:
                print 'wring detected SF type:', dict_list[0].get('Type')
                exit(0)
    return       new_dict_list

def ensemble_json():
    # -rw-rw-r-- 1 wenpeng wenpeng   663948 Jul 11 00:31 il9_system_output_concMT_BBN_NI_epoch4.json
    # -rw-rw-r-- 1 wenpeng wenpeng   467606 Jul 10 20:55 il9_system_output_noMT_BBN_NI_epoch4.json
    # -rw-rw-r-- 1 wenpeng wenpeng   813210 Jul  3 10:47 il9_system_output_noMT_epoch3.json
    # -rw-rw-r-- 1 wenpeng wenpeng   888750 Jul  3 10:50 il9_system_output_noMT_epoch4.json
    # -rw-rw-r-- 1 wenpeng wenpeng   338746 Jul 11 00:00 il9_system_output_onlyMT_BBN_epoch4.json
    root = '/save/wenpeng/datasets/LORELEI/il9/'
    json_list = [
                'il9_system_output_noMT_epoch4.json',
                'il9_system_output_noMT_BBN_NI_epoch4.json',
                'il9_system_output_onlyMT_BBN_epoch4.json',
                'il9_system_output_concMT_BBN_NI_epoch4.json']
    overall_list = []
    key_count = defaultdict(int)
    for json_file in json_list:
        json_i = json.load(codecs.open(root+json_file, 'r', 'utf-8'))
        print 'size:',len(json_i)
        overall_list+=json_i

        for dic in json_i:
            doc_id = dic.get('DocumentID')
            type = dic.get('Type')
            kb_id = dic.get('Place_KB_ID')
            key = (doc_id, type, kb_id)
            key_count[key]+=1
    keys = key_count.keys()
    print 'before key_count size:', len(key_count)
    for key in keys:
        if key_count.get(key) < len(json_list)/2:
            del key_count[key]
    print 'after key_count size:', len(key_count)

    new_dict_list = de_duplicate(overall_list, key_count)
    writefile = codecs.open(root+'il9_system_output_ensemble.json' ,'w', 'utf-8')
    print 'final size:', len(new_dict_list)
    json.dump(new_dict_list, writefile)
    writefile.close()
    print 'ensemble over'

def majority_ele_in_list(lis):
    c = Counter(lis)
    return c.most_common()[0][0]

def concate_json():
    # il='/save/wenpeng/datasets/LORELEI/il9/il9_system_output_noMT_epoch3.json'
    # eng= '/save/wenpeng/datasets/LORELEI/il9il10-eng/il9il10-eng_system_output_epoch3.json'
    # il_data = json.load(codecs.open(il, 'r', 'utf-8'))
    # eng_data = json.load(codecs.open(eng, 'r', 'utf-8'))
    # conb = il_data+eng_data
    # writefile = codecs.open('/save/wenpeng/datasets/LORELEI/il9/il9_system_output_final_epoch3.json' ,'w', 'utf-8')
    # json.dump(conb, writefile)
    # writefile.close()
    # il='/save/wenpeng/datasets/LORELEI/il10/il10_system_output_noMT_epoch4.json'
    # eng= '/save/wenpeng/datasets/LORELEI/il10-eng/il10-eng_system_output_epoch4.json'

    # il='/save/wenpeng/datasets/LORELEI/il9/il9_system_output_noMT_BBN_NI_epoch4.json'
    # eng= '/save/wenpeng/datasets/LORELEI/il9-eng/il9-eng_system_output_epoch4.json'

    il='/save/wenpeng/datasets/LORELEI/il10/il10_system_output_noMT_BBN_NI_epoch4.json'
    eng= '/save/wenpeng/datasets/LORELEI/il10-eng/il10-eng_system_output_epoch4.json'


    il_data = json.load(codecs.open(il, 'r', 'utf-8'))
    eng_data = json.load(codecs.open(eng, 'r', 'utf-8'))
    conb = il_data+eng_data
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/il10/il10_il_and_eng_bbn_ni_epoch4.json' ,'w', 'utf-8')
    json.dump(conb, writefile)
    writefile.close()
if __name__ == '__main__':
    # concate_json()
    ensemble_json()
