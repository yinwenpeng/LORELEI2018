import pickle
import codecs
from preprocess_common import IL_into_test_filteredby_NER_2018,load_EDL2018_output




if __name__ == '__main__':
    docid2entity_pos_list = load_EDL2018_output('/save/wenpeng/datasets/LORELEI/il10/il10_setE-anno_cand_v3.tab')
    IL_into_test_filteredby_NER_2018('/save/wenpeng/datasets/LORELEI/il10/monolingual_text/','/save/wenpeng/datasets/LORELEI/il10/il10-setE-as-test-input_ner_filtered', docid2entity_pos_list, 2)

    # json_validation('/save/wenpeng/datasets/LORELEI/il9/il9_system_output_forfun_w2.json')
