

import json
import codecs




def comb_json():
    # il='/save/wenpeng/datasets/LORELEI/il9/il9_system_output_noMT_epoch3.json'
    # eng= '/save/wenpeng/datasets/LORELEI/il9il10-eng/il9il10-eng_system_output_epoch3.json'
    # il_data = json.load(codecs.open(il, 'r', 'utf-8'))
    # eng_data = json.load(codecs.open(eng, 'r', 'utf-8'))
    # conb = il_data+eng_data
    # writefile = codecs.open('/save/wenpeng/datasets/LORELEI/il9/il9_system_output_final_epoch3.json' ,'w', 'utf-8')
    # json.dump(conb, writefile)
    # writefile.close()
    il='/save/wenpeng/datasets/LORELEI/il10/il9_system_output_noMT_epoch3.json'
    eng= '/save/wenpeng/datasets/LORELEI/il9il10-eng/il9il10-eng_system_output_epoch3.json'
    il_data = json.load(codecs.open(il, 'r', 'utf-8'))
    eng_data = json.load(codecs.open(eng, 'r', 'utf-8'))
    conb = il_data+eng_data
    writefile = codecs.open('/save/wenpeng/datasets/LORELEI/il9/il9_system_output_final_epoch3.json' ,'w', 'utf-8')
    json.dump(conb, writefile)
    writefile.close()
if __name__ == '__main__':
    comb_json()
