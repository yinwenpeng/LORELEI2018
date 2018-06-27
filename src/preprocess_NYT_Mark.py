
import xml.etree.ElementTree as ET
import json
import codecs



type2label_id = {'crimeviolence':8, 'med':3, 'search':4, 'food':1, 'out-of-domain':9, 'infra':2, 'water':7, 'shelter':5,
'regimechange':10, 'evac':0, 'terrorism':11, 'utils':6}

# -rw-rw-r-- 1 mssammon cs_danr 61000 Jun 17 17:54 crimeviolence.txt_mrg.txt
# -rw-rw-r-- 1 mssammon cs_danr 61000 Jun 17 17:55 disasters.txt_mrg.txt
# -rw-rw-r-- 1 mssammon cs_danr 61000 Jun 17 17:54 distractor.txt_mrg.txt
# -rw-rw-r-- 1 mssammon cs_danr 61000 Jun 17 17:53 evac.txt_mrg.txt
# -rw-rw-r-- 1 mssammon cs_danr 61000 Jun 17 17:52 food.txt_mrg.txt
# -rw-rw-r-- 1 mssammon cs_danr 61000 Jun 17 17:53 infra.txt_mrg.txt
# -rw-rw-r-- 1 mssammon cs_danr 61000 Jun 17 17:53 med.txt_mrg.txt
# -rw-rw-r-- 1 mssammon cs_danr 61000 Jun 17 17:54 regimechange.txt_mrg.txt
# -rw-rw-r-- 1 mssammon cs_danr 61000 Jun 17 17:52 search.txt_mrg.txt
# -rw-rw-r-- 1 mssammon cs_danr 61000 Jun 17 17:54 shelter.txt_mrg.txt
# -rw-rw-r-- 1 mssammon cs_danr 61000 Jun 17 17:54 terrorism.txt_mrg.txt
# -rw-rw-r-- 1 mssammon cs_danr 61000 Jun 17 17:51 util.txt_mrg.txt
# -rw-rw-r-- 1 mssammon cs_danr 61000 Jun 17 17:53 water.txt_mrg.txt
def preprocess_nyt():
    folder = '/shared/experiments/mssammon/lorelei/2018/keywords-for-sf/nyt-docs-from-keywords-origmspluscb/'
    filelist=['evac.txt_mrg.txt',
    'food.txt_mrg.txt',
    'infra.txt_mrg.txt',
    'med.txt_mrg.txt',
    'search.txt_mrg.txt',
    'shelter.txt_mrg.txt',
    'util.txt_mrg.txt',
    'water.txt_mrg.txt',
    'crimeviolence.txt_mrg.txt',
    'distractor.txt_mrg.txt',
    'regimechange.txt_mrg.txt',
    'terrorism.txt_mrg.txt'
    ]

    id2label = {y:x for x,y in type2label_id.iteritems()}
    writefile = codecs.open('/home/wyin3/Datasets/NYT-Mark-top10-id-label-text.txt', 'w', 'utf-8')
    co = 0
    for id, fil in enumerate(filelist):
        readfile_l1 = codecs.open(folder+fil, 'r', 'utf-8')
        limit=0
        for line_fil in readfile_l1:
            print 'reading .....', fil, line_fil.strip()
            readfile_l2 = json.load(codecs.open(line_fil.strip(), 'r', 'utf-8'))
            token_list = readfile_l2.get('tokens')
            label = id2label.get(id)
            if label is None:
                print 'label is None:'
                exit(0)
            writefile.write(str(id)+'\t'+label+'\t'+' '.join(token_list)+'\n')
            limit+=1
            if limit == 10:
                break
            co+=1
            if co % 500 == 0:
                print 'co...', co
        readfile_l1.close()
    print 'over'



if __name__ == '__main__':
    preprocess_nyt()
