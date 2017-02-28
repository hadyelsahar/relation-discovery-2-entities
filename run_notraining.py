__author__ = 'hadyelsahar'

import pandas as pd
from IPython.core.debugger import Tracer; debug_here = Tracer()
from sklearn.metrics import classification_report

from sklearn.decomposition import PCA
from evaluation.evaluation import ClusterEvaluation
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils.glove import GloveVectorizer
import numpy as np
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import argparse

class Run:
    def __init__(self, df, glovefile):
        """

        :param test_dataset:

        """
        self.glovefile = glovefile
        # selecting sentences with labels (a.k.a test set)

        df = df[df.relation.notnull()]

        # Initializing Vectorizer
        print "loading Train Dataset"

        # removing non parsable chars by nltk and scitkit learn
        df['sentence'] = df.apply(lambda x: x.sentence.decode('utf-8', 'ignore').encode("utf-8"), axis=1)

        # removing sentences longer than 500 words (bug in the provided datasets)
        df['length'] = df.apply(lambda x: len((word_tokenize(x.sentence))), axis=1)
        df = df[df.length < 500]

        self.data = df
        self.labels = df.relation.values


    def vectorize(self):
        """
        vectorize part of the input data
        :return: X
        """

        #{name:"", feature:featureVectors, PCA:True, PCA_S=10}
        self.features = []

        # Features Vectors:
        ###################
        # Sentence Features
        idf = TfidfVectorizer(binary=True)
        s_idf = idf.fit_transform(self.data['sentence'].values).toarray()

        # Sentence With Entities Replaced by Types
        # ss = self.data.apply(lambda x: x['sentence'].replace(x['sub'], x['type'].split('-')[0]).replace(x['obj'], x['type'].split('-')[1]), axis=1)
        # countvectorizer = CountVectorizer()
        # ss_count = countvectorizer.fit_transform(ss).toarray()
        # tfidf = TfidfVectorizer()
        # ss_tfidf = tfidf.fit_transform(ss).toarray()
        # idf = TfidfVectorizer(binary=True)
        # ss_idf= idf.fit_transform(ss).toarray()

        # Sentence Glove Word2Vec
        glove_vectorizer = GloveVectorizer(self.glovefile)
        glove_vectorizer.fit(self.data['sentence'].values)
        # w2v = glove_vectorizer.transform_sumembed(self.data['sentence'].values, average=True)
        w2v = glove_vectorizer.transform_sumembed(self.data['sentence'].values, idf=True)

        self.w2v = w2v
        ###### Adding Features ####
        self.features += [
            # {'name': "sentence_idf", 'feature': s_idf, 'PCA': True, 'PCA_size': 10},
            # {'name': "sentence_prep_count", 'feature': ss_count, 'PCA': True, 'PCA_size': 10},
            # {'name': "sentence_prep_tfidf", 'feature': ss_tfidf, 'PCA': True, 'PCA_size': 10},
            # {'name': "sentence_prep_idf", 'feature': ss_idf, 'PCA': True, 'PCA_size': 10}
            {'name': "word2vec_sum", 'feature': w2v, 'PCA': False}
        ]

        tmp = []
        for f in self.features:
            if f['PCA']:
                pca = PCA(n_components=f['PCA_size'])
                tmp.append(pca.fit_transform(f['feature']))
            else:
                tmp.append(f['feature'])

        X = np.hstack(tmp)
        # print "returning dense"
        self.vectorizeddata = X

        return self.vectorizeddata

    def cluster(self, x, clustering=None, n_clusters=6):

        # print "Starting Kmeans clustering.."
        if clustering is None:
            clustering = KMeans(n_clusters=n_clusters)

        predictions_minibatch = clustering.fit_predict(x)
        # print "done Kmeans clustering.."
        self.clusters = predictions_minibatch

        e = ClusterEvaluation(self.labels, predictions_minibatch)
        m = e.printEvaluation()

        return m

    # def tune_clustering(self):
    #     f1s = range(5, 10, 1) + [1]
    #     f2s = range(8, 12, 1) + [0]
    #     f3s = range(5, 10, 1) + [0]
    #     f4s = range(50, 300, 50)
    #     f5s = range(50, 300, 50)
    #     # f1s = [0]
    #     # f2s = [0]
    #     # f3s = [0]
    #     # f4s = [0]
    #     # f5s = [0]
    #
    #     results = []
    #     self.results = []
    #     maxf = 0
    #     for f1, f2, f3, f4, f5 in list(itertools.product(f1s, f2s, f3s, f4s, f5s)):
    #
    #         self.vectorize(f1, f2, f3, f4, f5)
    #         fscore = self.cluster(self.vectorizeddata, AgglomerativeClustering, n_clusters=5)['Elementwise B3 F1']
    #         results.append([fscore, f1, f2, f3, f4, f5])
    #         self.results = results
    #         if fscore > maxf:
    #             maxf = fscore
    #             print "%s %s %s %s %s --> %s" % (f1, f2, f3, f4, f5, fscore)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='this file is for building no training features for diego et al. dataset and be able to classify them')
    parser.add_argument('-i', '--input', help="dataset file")
    parser.add_argument('-g', '--glovefile', help="glove file")
    parser.add_argument('-n', '--nclusters', help="glove file")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    r = Run(df, args.glovefile)
    r.vectorize()
    m = r.cluster(r.vectorizeddata, n_clusters=int(args.nclusters))
    print m



