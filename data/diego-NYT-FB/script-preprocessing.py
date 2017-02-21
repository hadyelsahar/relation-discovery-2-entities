import argparse
import os
import pandas as pd
import spotlight
from multiprocessing import Pool
import numpy as np

num_partitions = 10     # number of partitions to split dataframe
num_cores = 8           # number of cores on your machine


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

parser = argparse.ArgumentParser(description='preprocessing script to add subj and obj entities urls and for diego et al dataset')
parser.add_argument('-i', '--input', help='folder containing all google relation extraction files', required=True)
parser.add_argument('-o', '--out', help='outfile', required=True)
args = parser.parse_args()


df = pd.read_csv(args.input, sep='\t', names=['dep', 'sub', 'obj', 'type', 'trigger', 'file', 'sentence', 'pos', 'relation'])

types_dict = {
    "LOCATION": ["DBpedia:Location","DBpedia:PopulatedPlace", "DBpedia:Place"],
    "PERSON": ['DBpedia:Person'],
    "ORGANIZATION": ['DBpedia:Organisation']
}

SPOTLIGHT_URL = 'http://localhost:2222/rest/annotate'
SPOTLIGHT_CONF = 0.3
SPOTLIGHT_SUPPORT = 1
COUNTER = 0
SAVE_EVERY = 1000


def get_entities(x):
    global df
    global COUNTER

    if COUNTER % 10 == 0:
        print "%s documents tagged" % COUNTER

    COUNTER += 1

    sub = {
        "uri": None,
        "type": None,
        "offset": None
    }
    obj = {
        "uri": None,
        "type": None,
        "offset": None
    }

    # sub_type, obj_type = x.type.split("-")[0]
    # entities = []
    # types = []

    try:
        # shorten sentence to speedup.

        es = spotlight.annotate(SPOTLIGHT_URL, x['sub'], SPOTLIGHT_CONF, SPOTLIGHT_SUPPORT)
        eo = spotlight.annotate(SPOTLIGHT_URL, x['obj'], SPOTLIGHT_CONF, SPOTLIGHT_SUPPORT)
        entities = es + eo

        for e in entities:
            if e["surfaceForm"] == x['sub']:
                sub['uri'] = e['URI']
                sub['type'] = [i for i in e["types"].split(',') if i.startswith("DBpedia")]
                sub['offset'] = e['offset']

            if e["surfaceForm"] == x['obj']:
                obj['uri'] = e['URI']
                obj['type'] = [i for i in e["types"].split(',') if i.startswith("DBpedia")]
                obj['offset'] = e['offset']

    except Exception as e:
        print e.message

    if sub['type'] is None and x.type.split("-")[0] in types_dict:
        sub['type'] = types_dict[x.type.split("-")[0]]

    if obj['type'] is None and x.type.split("-")[1] in types_dict:
        obj['type'] = types_dict[x.type.split("-")[1]]

    return sub['uri'], sub['type'], sub['offset'], obj['uri'], obj['type'], obj['offset']


def add_entities(df):
    entities = lambda x: pd.Series(get_entities(x))
    df[['sub_uri', 'sub_type', 'sub_offset', 'obj_uri', 'obj_type', 'obj_offset']] = df.apply(entities, axis=1)
    return df

# df[['sub_uri', 'sub_type', 'sub_offset', 'obj_uri', 'obj_type', 'obj_offset']] = df.apply(entities, axis=1)
df = parallelize_dataframe(df, add_entities)
df.to_csv(args.out, encoding='utf-8')

st = df[df['sub_uri'].notnull()][df['obj_uri'].notnull()].shape[0]
print "%s out of %s .. (%s percent) of the documents has been sucessfully tagged" % (st, df.shape[0], float(st)/df.shape[0])
