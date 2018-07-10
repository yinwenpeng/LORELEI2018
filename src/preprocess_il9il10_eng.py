import pickle
import codecs
from preprocess_common import IL_eng_into_test_filteredby_NER_2018,load_EDL2018_output




if __name__ == '__main__':
    IL_eng_into_test_filteredby_NER_2018('/save/wenpeng/datasets/LORELEI/il9il10-eng/monolingual_text/','/save/wenpeng/datasets/LORELEI/il9il10-eng/il9il10-eng-setE-as-test-input_ner_filtered', 2)

    # json_validation('/save/wenpeng/datasets/LORELEI/il9/il9_system_output_forfun_w2.json')
