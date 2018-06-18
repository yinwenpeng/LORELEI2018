

import json
import codecs


def denoise_text(list_of_list_words):
    sents = []
    for lis in list_of_list_words:
        sent = ' '.join(lis)
        sents.append(sent)
    return ' '.join(sents)


def reformat_E30():
    '''
    {u'med': 3, u'search': 4, u'food': 1, u'utils': 6, u'infra': 2, u'water': 7, u'shelter': 5, u'evac': 0}
    '''
    read_label_file = open('/save/wenpeng/datasets/LORELEI/translated2017/E30-needs.tsv', 'r')
    endID2typelist={}
    type2id = {'med': 3, 'search': 4, 'food': 1, 'utils': 6, 'infra': 2, 'water': 7, 'shelter': 5, 'evac': 0}
    line_co=0
    for line in read_label_file:
        line_co+=1
        if line_co>1:
            parts=line.strip().split('\t')
            entID = parts[4]
            typ = parts[3]
            if endID2typelist.get(entID) is  None:
                endID2typelist[entID] = [typ]
            else:
                old_type_list = endID2typelist.get(entID)
                old_type_list.append(typ)
                endID2typelist[entID] = old_type_list


    read_label_file.close()
    # print endID2typelist
    # exit(0)


    # exit(0)
    E30_data = json.load(codecs.open('/save/wenpeng/datasets/LORELEI/translated2017/E30-human.json', 'r', 'utf-8'))
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/translated2017/E30_id_label_segment.txt', 'w', 'utf-8')
    size = len(E30_data)
    print 'doc size:', size
    valid_doc = 0
    valid_seg = 0
    for i in range(size):
        doc_id = E30_data[i].get('id')
        # print 'doc_id:', doc_id
        # print 'E30_data[i]:', E30_data[i]
        place_list = E30_data[i].get('places')
        if len(place_list) ==0:
            continue
        else:
            valid_doc+=1
            sent_list =  E30_data[i]['translation']
            for ent_info  in place_list:
                # print 'place_list:', place_list
                # print 'ent_info:', ent_info
                sent_id = ent_info.get('sentence')
                ent_id = ent_info.get('entity_id')
                typelist_exp = endID2typelist.get(ent_id)
                if typelist_exp is None:
                    continue
                else:
                    valid_seg+=1
                    typeid_list = [str(type2id.get(typename)) for typename in typelist_exp]
                    segment_sent_ids = [sent_id-1, sent_id, sent_id+1]
                    refined_segment_sent_ids = [x for x in segment_sent_ids if x>=0 and x < len(sent_list)]

                    segment_wordlist =  [word for idd in refined_segment_sent_ids for word in sent_list[idd]]
                    segment_text = ' '.join(segment_wordlist)
                    writefile.write(' '.join(typeid_list)+'\t'+' '.join(typelist_exp)+'\t'+segment_text+'\n')

    writefile.close()
    print 'all done, valid_doc:', valid_doc, valid_seg







if __name__ == '__main__':
    reformat_E30()
