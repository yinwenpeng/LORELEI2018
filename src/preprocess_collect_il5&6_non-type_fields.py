import codecs
from preprocess_il5 import load_il5

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

def generate_entity_focused_trainingset(docid2text, docid2trans,docid2issue, docid2mention, docid2need):
    '''
    monolingual_text: doc_id: sent_list:['...','...'], boundary_list:[(1,12),(14,23)...]
    ground truth:
        issue: doc_id:[{'entity_id': place_id, 'frame_type': issue, 'issue_type':crimeviolence},{'entity_id':...}]
        mentions: doc_id:[{'entity_id': place_id, 'entity_type': GPE, 'start_char':12,'end_char':15}]
        needs: doc_id:[{'entity_id': place_id, 'frame_type': need, 'need_type':med, 'need_status': current, 'urgency_status': true/false, 'resolution_status':insufficient},{'entity_id':...}]
    '''
    type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2, 'water':7, 'shelter':5,
    'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/il5_translated_seg_level_as_training_all_fields.txt', 'w', 'utf-8')
    write_size = 0
    doc_union_issue_and_needs = set(docid2issue.keys())| set(docid2need.keys())
    for doc_id, doc_instance in docid2text.iteritems():
        sent_list = doc_instance.get('sent_list')
        trans_v1_sent_list = docid2trans[doc_id].get('sent_list_version1')
        trans_v2_sent_list = docid2trans[doc_id].get('sent_list_version2')
        boundary_list = doc_instance.get('boundary_list')

        if doc_id  in doc_union_issue_and_needs: #this doc has SF type labels
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
                    writefile.write(' '.join(map(str, iddlist))+'\t'+' '.join(labelstrlist_delete_duplicate)+'\t'+sent1.strip()+'\n')
                if len(sent2) > 0:
                    writefile.write(' '.join(map(str, iddlist))+'\t'+' '.join(labelstrlist_delete_duplicate)+'\t'+sent2.strip()+'\n')
                write_size+=1
    writefile.close()
    print 'write_size:', write_size


if __name__ == '__main__':
    docid2text, docid2issue, docid2mention, docid2need = load_il5()
    docid2trans = load_il5_translated()
