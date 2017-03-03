import numpy as np
import operator
from collections import defaultdict
from nltk.tokenize import word_tokenize
import re

class DepAttentionVectorizer:

    def depparse_tolist(self, s):
        stoplist = [u'prep', u'pobj', u'appos', u'nsubj', u'nn', u'dobj', u'partmod', u'poss']
        l = re.split("->|-<", s)
        return [i for i in l if i not in stoplist]

    def transform(self, X, D, seq_length=None):
        """
        :param X: list of sentences
        :param D: list of dependency parse between two sentences
        :return:
        """

        V = []
        for i, x in enumerate(X):
            deplist = self.depparse_tolist(D[i])
            tokens = word_tokenize(x)

            if seq_length is not None:
                v = np.zeros(seq_length)
            else :
                v = np.zeros(len(tokens))

            for j, w in enumerate(tokens):
                if w in deplist:
                    v[j] = 1
            V.append(v)

        return np.array(V)


