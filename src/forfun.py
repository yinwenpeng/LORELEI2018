import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time
import math
import codecs
import numpy as np
import json

def load_fasttext_word2vec_given_file(vec):


    writefile = codecs.open('system_output.json' ,'w', 'utf-8')
    lis=[]
    for i in range(100):
    	dic={}
    	dic['haha']='yes'
    	dic['val'] = vec[i]
    	lis.append(dic)
    json.dump(lis, writefile)
    writefile.close()

# ruby LoReHLT_Frame_Scorer.rb -p LoReHLT17 -L /save/wenpeng/datasets/LORELEI/il5_setE_monolingual_text/ltf/ -R /save/wenpeng/datasets/LORELEI/il5_unseq/setE/data/annotation/situation_frame/ -s /save/wenpeng/datasets/LORELEI/il5_system_output_forfun.json -f /save/wenpeng/datasets/LORELEI/official_scorer/reflist.txt -F /save/wenpeng/datasets/LORELEI/official_scorer/reflist.txt -o /save/wenpeng/datasets/LORELEI/official_scorer/Scores

ruby LoReHLT_Frame_Scorer.rb -p LoReHLT17 -L /shared/corpora/corporaWeb/lorelei/evaluation-20170804/il5/setE/data/monolingual_text/ltf/ -R /shared/corpora/corporaWeb/lorelei/evaluation-20170804/il5_unseq/setE/data/annotation/situation_frame/ -s /home/wyin3/LORELEI/il5_system_output_forfun.json -f /home/wyin3/LORELEI/official_scorer/reflist.txt -F /home/wyin3/LORELEI/official_scorer/reflist.txt -o /home/wyin3/LORELEI/official_scorer/Scores

ruby LoReHLT_Frame_Scorer.rb -p LoReHLT17 -L /Users/yinwenpeng/Downloads/LORELEI/ltf/ltf -R /Users/yinwenpeng/Downloads/LORELEI/situation_frame/situation_frame/ -s /Users/yinwenpeng/Downloads/LORELEI/il5_system_output_forfun.json -f /Users/yinwenpeng/Downloads/LORELEI/wp_filelist.txt -F /Users/yinwenpeng/Downloads/LORELEI/annotated_filelist_SF.tab -o /Users/yinwenpeng/Downloads/LORELEI/Scores


{"Confidence": 0.4801243543624878, "DocumentID": "IL9_NW_020576_20180503_I0040RL1P", "Justification_ID": "segment-5", "Place_KB_ID": "49518", "Resolution": "insufficient", "Status": "current", "Type": "shelter", "Urgent": false},
{"Confidence": 0.4801243543624878, "DocumentID": "IL9_NW_020576_20180503_I0040RL1P", "Justification_ID": "segment-5", "Place_KB_ID": "433561", "Resolution": "insufficient", "Status": "current", "Type": "shelter", "Urgent": false}

gem install nokogiri -v '1.8.3' --source 'https://rubygems.org/
if __name__ == '__main__':
    load_fasttext_word2vec_given_file(range(100))
