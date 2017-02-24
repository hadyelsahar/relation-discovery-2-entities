__author__ = 'hadyelsahar'

import pandas as pd
from IPython.core.debugger import Tracer;
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

debug_here = Tracer()
from sklearn.decomposition import PCA
from evaluation.evaluation import ClusterEvaluation
from sklearn import svm
import itertools
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.glove import Glove


class Run:
    def __init__(self, df):
        """

        :param test_dataset:
        """

        # Initializing Vectorizer

        print "loading Train Dataset"
        self.data = df


    def vectorize(self, f1=4, f2=4, f3=10, f4=100, f5=100):
        """
        vectorize part of the input data
        :return: X
        """

        # Features Vectors:
        ###################
        # Sentence Features
        countvectorizer = CountVectorizer()
        sent_count = countvectorizer.fit_transform(self.data['sentence'].values).toarray()
        tfidf = TfidfVectorizer()
        sent_tfidf = tfidf.fit_transform(self.data['sentence'].values).toarray()
        idf = TfidfVectorizer(binary=True)
        sent_count = idf.fit_transform(self.data['sentence'].values).toarray()
        # Sentence With Entities Replaced by Types
        ss = self.data.apply(lambda x: x['sentence'].replace(x['sub'],x['type'].split('-')[0]).replace(x['obj'], x['type'].split('-')[1]), axis=1)
        countvectorizer = CountVectorizer()
        ss_count = countvectorizer.fit_transform(ss).toarray()
        tfidf = TfidfVectorizer()
        ss_tfidf = tfidf.fit_transform(ss).toarray()
        idf = TfidfVectorizer(binary=True)
        ss_count = idf.fit_transform(ss).toarray()


        subjectnorm = np.array(self.data.subjectnorm.values.tolist())
        objectnorm = np.array(self.data.objectnorm.values.tolist())

        # attr = [x.replace("in","").replace("to","").replace("on","").replace("at","") for x in self.data.A.values]
        w2v = self.word2vecvectorizer.fit_transform(self.data.A.values)
        w2v = np.array([np.average(j, axis=0) for j in w2v])
        np.random.seed(3)

        # TFIDF Weighted sum of word vectors:
        tfidfvec = TfidfVectorizer()

        tfidfvec.fit(self.all_attributes)
        tfidf = tfidfvec.fit_transform(self.data.A.values)

        ww2v = np.zeros((self.data.shape[0], self.word2vecvectorizer.model.vector_size))
        tfidf_feature_names = tfidfvec.get_feature_names()
        tfidf_feature_w2v = np.array([self.word2vecvectorizer.word2vec(w) for w in tfidf_feature_names])
        for i, a in enumerate(tfidf.todense()):
            r = np.multiply(tfidf_feature_w2v, a.transpose())
            r = np.sum(r, axis=0) / len(self.data["tokens"].values[i])
            ww2v[i:] = r

        np.random.seed(3)

        # dirrand = {1:one, -1: negone}

        dir = np.array([i for i in self.data.direction.values.tolist()]).reshape([-1, 1])

        # has preposition #used for detection of direction

        pca = PCA(n_components=f1)
        pos = pca.fit_transform(pos)
        pca = PCA(n_components=f2)
        dep = pca.fit_transform(dep)

        dirtype = np.hstack((
            dir,
            subjectnorm,
            objectnorm,
        ))

        pca = PCA(n_components=f3)
        dirtype = pca.fit_transform(dirtype)

        X1 = np.hstack((
            pos,
            dep,
            dirtype
        ))

        pca = PCA(n_components=f4)
        w2v = pca.fit_transform(w2v)
        pca = PCA(n_components=f5)
        ww2v = pca.fit_transform(ww2v)

        X = np.hstack((
            X1,
            w2v,
            ww2v,
        ))

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

        e = ClusterEvaluation(self.data.label, predictions_minibatch)
        m = e.printEvaluation()

        return m

    def tune_clustering(self):
        f1s = range(5, 10, 1) + [1]
        f2s = range(8, 12, 1) + [0]
        f3s = range(5, 10, 1) + [0]
        f4s = range(50, 300, 50)
        f5s = range(50, 300, 50)
        # f1s = [0]
        # f2s = [0]
        # f3s = [0]
        # f4s = [0]
        # f5s = [0]

        results = []
        self.results = []
        maxf = 0
        for f1, f2, f3, f4, f5 in list(itertools.product(f1s, f2s, f3s, f4s, f5s)):

            self.vectorize(f1, f2, f3, f4, f5)
            fscore = self.cluster(self.vectorizeddata, AgglomerativeClustering, n_clusters=5)['Elementwise B3 F1']
            results.append([fscore, f1, f2, f3, f4, f5])
            self.results = results
            if fscore > maxf:
                maxf = fscore
                print "%s %s %s %s %s --> %s" % (f1, f2, f3, f4, f5, fscore)
