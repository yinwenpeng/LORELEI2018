__author__ = "Oleg Aulov (oleg.aulov@nist.gov)"
__version__ = "Development: 0.8"
__date__ = "05/14/2018"

import glob
import os
import sys
import numpy as np
import pandas as pd
import csv
import json
from pandas.io.json import json_normalize
import jsonschema
from jsonschema import Draft3Validator


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

def getreference(path, gravity):
    #path = # use your path
    allFiles = glob.glob(path + "/needs/*.tab") + glob.glob(path + "/issues/*.tab")
    #print(allFiles)
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0, sep='\t', quoting=csv.QUOTE_NONE, dtype={'doc_id': object, 'kb_id': object})
        list_.append(df)
    reference = pd.concat(list_, sort=True)
    reference['status'] = reference[['need_status','issue_status']].apply(lambda x: x['need_status'] if pd.isnull(x['issue_status']) else x['issue_status'], axis=1)
    reference['type'] = reference[['issue_type','need_type']].apply(lambda x: x['issue_type'] if pd.isnull(x['need_type']) else x['need_type'], axis=1)
    reference.drop(['need_status', 'issue_status', 'description', "issue_type", "need_type", "proxy_status", "reported_by", "resolved_by"], axis=1, inplace=True)
    reference = reference[reference["place_id"] != "none"]

    d = {'True': True, 'False': False, np.nan: False}
    reference['urgency_status'].replace(d, inplace = True)
    # reference['urgent'] = reference[['scope','severity']].apply(lambda x: True if x['scope'] > 1 and x['severity'] > 1 else False, axis=1)
    reference['urgent'] = reference['urgency_status']
    d = {'insufficient': True, 'sufficient': False, np.nan: False}
    reference['unresolved'] = reference['resolution_status'].map(d)

    d = {'current': True, 'past': False, "future": False, "not_current": False, np.nan: False}
    reference['current'] = reference['status'].map(d)


    reference['gravity'] = reference.apply(gravity, axis=1)
    reference['frame_count'] = 1
    return reference

def getsubmission(filename, gravity):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    schemafile = os.path.join(base_dir, "LoReHLT18_system-output-schema.json")
    try:
        with open(schemafile, 'r') as f:
            schema_data = f.read()
            schema = json.loads(schema_data)
    except:
        sys.exit('CAN NOT OPEN JSON SCHEMA FILE: '+ schemafile)

    try:
        with open(filename) as f:
            d = json.load(f)
    except:
        sys.exit('CAN NOT OPEN JSON SUBMISSION FILE: '+ schemafile)

    try:
        # v = Draft6Validator(schema)
        # errors = sorted(v.iter_errors(d), key=lambda e: e.path)
        # for error in errors:
        #     for suberror in sorted(error.context, key=lambda e: e.schema_path):
        #         print(list(suberror.schema_path), suberror.message, sep=", ")
        # v = Draft3Validator(schema)
        # for error in sorted(v.iter_errors(d), key=str):
        #     print(error.message)
        jsonschema.validate(d, schema)
    except:
        sys.exit('JSON SUBMISSION FILE FAILED VALIDATION')

    mysubmission = json_normalize(d)

    d = {True: True, False: False, np.nan: False}
    mysubmission['urgent'] = mysubmission['Urgent'].map(d)
    mysubmission.drop('Urgent', axis=1, inplace=True)

    d = {'insufficient': True, 'sufficient': False, np.nan: False}
    mysubmission['unresolved'] = mysubmission['Resolution'].map(d)

    d = {'current': True, 'past': False, "future": False, "not_current": False, np.nan: False}
    mysubmission['current'] = mysubmission['Status'].map(d)

    mysubmission['gravity'] = mysubmission.apply(gravity, axis=1)
    mysubmission['frame_count'] = 1
    return mysubmission


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def gain(row):
    if row['gravity'] >= 3:
        return 5
    elif 2 <= row['gravity'] < 3 :
        return 3
    else:
        return 1


def gravity(row):
    if row['current'] and row['unresolved'] and row['urgent']:
        return True
    else:
        return False


# precision at N
def precisionN(KBframe, my_n):
    KBframe["correct"] = KBframe.gain > 0
    KBframe = KBframe[KBframe.gravity > 0]
    if KBframe.shape[0] > my_n:
        KBframe = KBframe.head(my_n)
    return KBframe[KBframe.correct == True].shape[0] / KBframe.shape[0]

def genScore(row):
    if row["doc_id"] == row["DocumentID"]:
        return 1
    elif pd.notnull(row["doc_id"]) and pd.isnull(row["DocumentID"]):
        return 1
    else:
        return 0

#MAP
def mapmar(reference, system, eqclass, threshold = 0):
    avgPrecision = []
    recall = []

    grouped = reference.groupby(["type","kb_id"])

    for KBsituation in grouped:
        refKBsituation = KBsituation[1].groupby(eqclass).size().to_frame('weight').reset_index()
        sysKBsituation = system[(system["Place_KB_ID"] == str(KBsituation[1]["kb_id"].unique()[0])) & \
                                (system["Type"] == str(KBsituation[1]["type"].unique()[0]))]

        KBmerged = pd.merge(refKBsituation, sysKBsituation, how='outer', left_on=[ "doc_id"], right_on = ['DocumentID'])\
        .sort_values(by='Confidence', ascending=False, na_position='last')\
        .drop_duplicates(subset = eqclass + ['DocumentID'], keep='first')
        KBmerged['score'] = KBmerged.apply(genScore, axis=1)
        KBmergedAP = KBmerged[np.isfinite(KBmerged['Confidence'])]
        KBmerged.Confidence.fillna(-1, inplace=True)
        KBmerged.weight = KBmerged.weight.fillna(1).astype('int64')
        KBmerged = KBmerged.loc[KBmerged.index.repeat(KBmerged['weight'])]

        if not KBmergedAP.empty:
            avgPrecision.append(average_precision_score(KBmergedAP.score, KBmergedAP.Confidence))
        else:
            avgPrecision.append(0.0)
        recall.append(recall_score(KBmerged.score, (KBmerged.Confidence>threshold).astype(int)))
    meanAP = np.mean(avgPrecision)
    macroAR = np.mean(recall)
    return meanAP, macroAR

def genNDCGplot(k, myndcg, outdir, sysname):
    plt.scatter(k, myndcg, marker='.',s=10, linewidth=0.5, color = "red")
    #plt.plot(k, myndcg, linestyle='-', color = "blue")
    plt.title("nDCG@k for " + sysname)
    plt.xlabel('k')
    plt.ylabel('nDCG')
    plt.grid()
    gc = plt.gcf()
    gc.set_size_inches(7, 7)
    #plt.legend(["dfbdfbdbdbd"], loc='upper right')
    pp = PdfPages(os.path.join(outdir,'curve_nDCG' +'.pdf'))
    pp.savefig(plt.gcf())
    pp.close()
    plt.close()
    return

def getreferenceNDCG(reference):
    refNDCG = reference.groupby(["kb_id","type"])["urgent","unresolved","current","gravity", "frame_count"].agg(['sum']).reset_index()
    refNDCG.columns = refNDCG.columns.droplevel(1)
    refNDCG = refNDCG.sort_values(by=['gravity', "frame_count"], ascending = False)
    refNDCG['gain'] = refNDCG.apply(gain, axis=1)
    return refNDCG

def genNDCG(referenceNDCG, systemTable):
    mysystem = systemTable.groupby(["Place_KB_ID","Type"])["urgent","unresolved","current","gravity", "frame_count"].agg(['sum']).reset_index()
    mysystem.columns = mysystem.columns.droplevel(1)
    mysystem = mysystem.sort_values(by=['gravity', "frame_count"], ascending = False)
    merged = pd.merge(mysystem, referenceNDCG[["kb_id", "type","gain"]], how='left', left_on=['Place_KB_ID','Type'], right_on = ['kb_id','type'])
    merged.gain = merged.gain.fillna(0)
    return [ndcg_at_k(merged['gain'], i) for i in range(1, len(referenceNDCG['gain']))]
