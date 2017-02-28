import numpy as np
import operator
from collections import defaultdict
from nltk.tokenize import word_tokenize



class TypeVectorizer:

    def __init__(self):

        self.types = None
        self.types_dict = None
        self.OOT = "OOT" # out of types

    def fit(self, X):
        """
        :param X: list of inputs for  each input is 2 dimensional (entity may have multiple types)
        :return:
        """

        self.types = list(set([t for ts in X for t in ts]))   # flatten all arrays and take unique

        self.types.append(self.OOT)
        self.types_dict = dict(zip(self.types, range(0, len(self.types))))

    def fit_transform_onehot(self, X):
        """
        :param X:  list of list of types i.e. every item can have multiple labels
        :return: 1 hot vector for ones in front of the provided labels
        """

        self.fit(X)

        v = np.zeros((X.shape[0], len(self.types)), dtype=int)

        for i, x in enumerate(X):
            for t in x:
                if t in self.types_dict:
                    v[i, self.types_dict[t]] = 1
                else:
                    v[i, self.types_dict[self.OOT]] = 1

        return v







