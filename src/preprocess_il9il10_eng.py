import pickle
import codecs
from preprocess_common import IL_eng_into_test_filteredby_NER_2018,load_EDL2018_output,IL_into_test_filteredby_NER_2018




if __name__ == '__main__':
    # IL_eng_into_test_filteredby_NER_2018('/save/wenpeng/datasets/LORELEI/il9-eng/monolingual_text/','/save/wenpeng/datasets/LORELEI/il9-eng/il9-eng-setE-as-test-input_ner_filtered', 2)
    # IL_eng_into_test_filteredby_NER_2018('/save/wenpeng/datasets/LORELEI/il10-eng/monolingual_text/','/save/wenpeng/datasets/LORELEI/il10-eng/il10-eng-setE-as-test-input_ner_filtered', 2)

    # docid2entity_pos_list = load_EDL2018_output('/save/wenpeng/datasets/LORELEI/il9-eng/il9_cp2_english.nil.fix.tab')
    # IL_into_test_filteredby_NER_2018('/save/wenpeng/datasets/LORELEI/il9-eng/monolingual_text/','/save/wenpeng/datasets/LORELEI/il9-eng/il9-eng-setE-as-test-input_ner_filtered', docid2entity_pos_list, 2)

    docid2entity_pos_list = load_EDL2018_output('/save/wenpeng/datasets/LORELEI/il10-eng/il10_cp2_english.nil.fix.tab')
    IL_into_test_filteredby_NER_2018('/save/wenpeng/datasets/LORELEI/il10-eng/monolingual_text/','/save/wenpeng/datasets/LORELEI/il10-eng/il10-eng-setE-as-test-input_ner_filtered', docid2entity_pos_list, 2)
