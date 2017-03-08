import numpy as np
import operator
from collections import defaultdict
from nltk.tokenize import word_tokenize
import re
from nltk.stem.snowball import EnglishStemmer


class DepAttentionVectorizer:

    def depparse_tolist(self, s):
        stoplist = [u'prep', u'pobj', u'appos', u'nsubj', u'nn', u'dobj', u'partmod', u'poss']
        l = re.split("->|-<", s)
        stemmer = EnglishStemmer()
        return [i.lower() for i in l if i not in stoplist]

    def transform(self, X, D, seq_length=None, C=1, C_inv=0):
        """
        :param X: list of sentences
        :param D: list of dependency parse between two sentences
        :return:
        """
        stemmer = EnglishStemmer()
        V = []
        for i, x in enumerate(X):
            deplist = self.depparse_tolist(D[i])
            tokens = word_tokenize(x)

            if seq_length is not None:
                v = np.ones(seq_length) * C_inv
            else :
                v = np.ones(len(tokens)) * C_inv

            for j, w in enumerate(tokens):

                if stemmer.stem(w.lower()) in deplist or w.lower() in deplist:
                    v[j] = C
            V.append(v)

        return np.array(V)


