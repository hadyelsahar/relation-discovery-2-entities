__author__ = 'hadyelsahar'

import json

import networkx as nx
import numpy as np
import requests


class CoreNlPClient:

    def __init__(self, serverurl="http://127.0.0.1:9000/", annotators=("tokenize", "ssplit", "pos", "lemma", "ner", "parse", "dcoref")):

        self.properties = {}
        self.annotators = annotators
        self.properties["annotators"] = ",".join(annotators)
        self.properties["tokenize.whitespace"] = False
        self.properties["outputFormat"] = "json"
        self.serverurl = serverurl

    def annotate(self, s):

        properties = json.dumps(self.properties)
        r = requests.post("%s?properties=%s" % (self.serverurl, properties), data=s.encode("utf-8"))

        if r.status_code == 200:
            x = json.loads(r.text, strict=False)

            return Parse(x, annotators=self.annotators)

        else:
            raise RuntimeError("%s \t %s" % (r.status_code, r.reason))


    def get_dates(self, s):

        dates = []
        entities = []

        parse = self.annotate(s)
        if 'DATE' in parse.ner:
            buf = []
            for tid, ner in enumerate(parse.ner):
                if ner == 'DATE':
                    buf.append(tid)
                elif len(buf) > 0:
                    dates.append(buf)
                    buf = []

            if len(buf) > 0:
                dates.append(buf)

        for d in dates:

            item = dict()
            item['URI'] = "DATE"
            item['surfaceForm'] = s[parse.positions[d[0]][0]:parse.positions[d[-1]][1]]
            item['offset'] = parse.positions[d[0]][0]
            item['types'] = "DATE"
            entities.append(item)

        return entities


    def get_entities(self, s):
        nertags = ["ORGANIZATION", "PERSON", "LOCATION", "DATE", "MISC"]
        e_positions = []
        entities = []

        parse = self.annotate(s)
        if any(t in parse.ner for t in nertags):
            for t in nertags:
                buf = []
                for tid, ner in enumerate(parse.ner):
                    if ner == t:
                        buf.append(tid)
                    elif len(buf) > 0:
                        e_positions.append(buf)
                        buf = []

                if len(buf) > 0:
                    e_positions.append(buf)

            for d in e_positions:

                item = dict()
                item['URI'] = parse.ner[d[0]]
                item['surfaceForm'] = s[parse.positions[d[0]][0]:parse.positions[d[-1]][1]]
                item['offset'] = parse.positions[d[0]][0]
                item['types'] = parse.ner[d[0]]
                entities.append(item)

        return entities





class Parse:
    """
    a class to hold the output of the corenlp parsed result
    """
    def __init__(self, parsed, annotators=None):
        """
        :param parsed: the output of invoking the stanford parser service
        :return
            tokens: list of tokens in text
            positions: tuples contains start and end offsets of every token
            postags: list of pos tags for every token
            ner: list of ner tags for every token
            parsed_tokens:
                    for every token list all incoming or out-coming relations
                    redundant but easy to call afterwards when writing rule based
                    {"in":[], "out":[]}] ... etc
        """

        self.tokens = [i['originalText'] for i in parsed["sentences"][0]["tokens"]]
        self.positions = [(i['characterOffsetBegin'], i['characterOffsetEnd']) for i in parsed["sentences"][0]["tokens"]]
        if "pos" in annotators:
            self.postags = [i['pos'] for i in parsed["sentences"][0]["tokens"]]
        if "ner" in annotators:
            self.ner = [i['ner'] for i in parsed["sentences"][0]["tokens"]]
        self.parsed_tokens = parsed["sentences"][0]["tokens"]

        if "parse" in annotators:
            # removing the root note and starting counting from 0
            self.dep = [{"in": [], "out":[]} for i in self.tokens]

            for d in parsed["sentences"][0]["collapsed-ccprocessed-dependencies"]:

                if d['dep'] == "ROOT":
                    self.dep[d['dependent']-1]["in"].append(("ROOT", None))

                else:
                    self.dep[d['dependent']-1]["in"].append((d['dep'], d['governor']-1))
                    self.dep[d['governor']-1]["out"].append((d['dep'], d['dependent']-1))

            # making graphs out of parses to make shortest path function
            self.depgraph = nx.MultiDiGraph()
            for tokid, dep in enumerate(self.dep):
                for iin in dep['in']:
                    self.depgraph.add_edge(iin[1], tokid, label=iin[0])

        if "coref" in annotators:
            self.corefs = parsed["corefs"]

        self.all = parsed


    def getshortestpath(self, source, target):

        try:
            path = nx.shortest_path(self.depgraph, source=source, target=target)

            s = ""

            for c, i in enumerate(path[:-1]):

                if c != 0:
                    s += " -> "
                    s += self.tokens[i]
                s += " -> "
                s += self.depgraph.get_edge_data(i, path[c+1])[0]['label']

            s += " -> "
            # s += self.tokens[path[-1]]

            return s

        except:
            return None


    def getchunks_using_patterns(self, patterns, sequence, removesubsets=True):
        """
        get chunks is a function that returns array of arrays containing chunks repreresenting specific patterns in the
        parsed sentence  ex : [[1, 2], [4, 5], [11, 12]]
        :param patterns: array of arrays representing patterns [['NN'], ['NN','NN','NP'],['VP', 'NP']]
        :param sequence: the sequence of tags to apply patterns on [self.ner, self.postags, self.tokens]
        :param inclusive: get all patterns matches regarding if they intersect with or not.
        :return:arrays containing chunks repreresenting specific patterns in the
        parsed sentence  ex : [[1, 2], [4, 5], [11, 12]]
        """

        allchunks = []

        # extraction of chunks that exists in patterns
        for pattern in patterns:
            starts = np.where(np.array(sequence) == pattern[0])[0]

            l = len(pattern)
            for start in starts:
                if sequence[start:start+l] == pattern:
                    allchunks.append(set(range(start, start+l)))

        # filteration of patterns and remove chunks who are subsets of other chunks keeping only the larger ones
        if not removesubsets:
            return allchunks

        else:
            chunkstoreturn = []
            for chunk1 in allchunks:

                is_subset_of_other = False

                for chunk2 in allchunks:
                    # no need to remove chunk1 because we are going to check for only subsets not equality
                    if chunk1 < chunk2:
                        is_subset_of_other = True
                        break

                if not is_subset_of_other:
                    chunkstoreturn.append(sorted(list(chunk1)))

            return chunkstoreturn















