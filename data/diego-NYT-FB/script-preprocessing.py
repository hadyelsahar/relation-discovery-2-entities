import argparse
import os
import pandas as pd
import spotlight
from multiprocessing import Pool
import numpy as np

num_partitions = 500     # number of partitions to split dataframe
num_cores = 2           # number of cores on your machine


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
entities = []

types_dict = {
    "LOCATION": ["DBpedia:Location", "DBpedia:PopulatedPlace", "DBpedia:Place"],
    "PERSON": ['DBpedia:Person'],
    "ORGANIZATION": ['DBpedia:Organisation']
}

SPOTLIGHT_URL = 'http://localhost:2222/rest/annotate'
SPOTLIGHT_CONF = 0.2
SPOTLIGHT_SUPPORT = 1
COUNTER = 1


def get_entities():
    global df
    global COUNTER
    global entities
    for index, x in df.iterrows():

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
            k = es + eo

            for e in k:
                if e["surfaceForm"] == x['sub']:
                    sub['uri'] = e['URI'].encode('utf-8')
                    sub['type'] = [i.encode('utf-8') for i in e["types"].split(',') if i.startswith("DBpedia") and i != "DBpedia:Agent"]
                    sub['offset'] = e['offset']

                if e["surfaceForm"] == x['obj']:
                    obj['uri'] = e['URI'].encode('utf-8')
                    obj['type'] = [i.encode('utf-8') for i in e["types"].split(',') if i.startswith("DBpedia") and i != "DBpedia:Agent"]
                    obj['offset'] = e['offset']

        except Exception as e:
            print e.message

        try:
            if sub['type'] is None and x.type.split("-")[0] in types_dict:
                sub['type'] = types_dict[x.type.split("-")[0]]

            if obj['type'] is None and x.type.split("-")[1] in types_dict:
                obj['type'] = types_dict[x.type.split("-")[1]]
        except Exception as e:
            print e.message

        entities.append((sub['uri'], sub['type'], sub['offset'], obj['uri'], obj['type'], obj['offset']))


# entities = lambda x: pd.Series(get_entities(x))
# entities = df.apply(entities, axis=1)
get_entities()
E = zip(*entities)
df['sub_uri'] = E[0]
df['sub_type'] = E[1]
df['sub_offset'] = E[2]
df['obj_uri'] = E[3]
df['obj_type'] = E[4]
df['obj_offset'] = E[5]

df.to_csv(args.out)

# df[['sub_uri', 'sub_type', sub_offset', 'obj_uri', 'obj_type', 'obj_offset']] = df.apply(entities, axis=1)
#df = parallelize_dataframe(df, add_entities)
# df = add_entities(df)
# df.to_csv(args.out, encoding='utf-8')

st = df[df['sub_uri'].notnull()][df['obj_uri'].notnull()].shape[0]
print "%s out of %s .. (%s percent) of the documents has been sucessfully tagged" % (st, df.shape[0], float(st)/df.shape[0])
