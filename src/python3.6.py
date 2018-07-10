import json
import jsonschema
import sys

def json_validation(outputfile):
    f= open('/save/wenpeng/datasets/LORELEI/official_scorer/LoReHLT18_SF_Scorer_0.8/LoReHLT18_system-output-schema.json', 'r')
    schema_data = f.read()
    schema = json.loads(schema_data)

    filevalid =  open(outputfile, 'r')
    d = json.load(filevalid)

    try:
        jsonschema.validate(d, schema)
        print('succeed')
    except:
        sys.exit('JSON SUBMISSION FILE FAILED VALIDATION')



if __name__ == '__main__':
    # docid2entity_pos_list = {}#load_EDL2017_output()
    # IL_into_test_filteredby_NER_2018('/save/wenpeng/datasets/LORELEI/il9/monolingual_text/','/save/wenpeng/datasets/LORELEI/il9/il9-setE-as-test-input_ner_filtered', docid2entity_pos_list, 2)

    json_validation('/save/wenpeng/datasets/LORELEI/il10/il10_system_output_noMT_epoch4.json')
