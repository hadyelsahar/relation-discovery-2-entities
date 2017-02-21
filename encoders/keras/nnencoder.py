'''This example demonstrates the use of Convolution1D for text classification.
Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, merge, Reshape, Lambda
from keras.layers import Embedding
from keras.layers import Input, Convolution1D, MaxPooling1D
from keras import backend as K
from sklearn.base import BaseEstimator, ClassifierMixin
import theano.tensor as T

class CNNDISTMULT(BaseEstimator, ClassifierMixin):

    def __init__(self, input_shape, features_size, conv_shape, embeddingsize=50, sentenceencodedsize=10, negative_sample=100, ent_nb=1000000, epochs=2500, batchsize=50, dropout=0.5):

        self.embeddingsize = embeddingsize
        self.sentenceencodedsize = sentenceencodedsize
        self.m, self.n = input_shape
        self.convshape = conv_shape
        self.negative_sample = negative_sample
        self.ent_nb = ent_nb

        # we add a Convolution1D, which will learn nb_filter
        # word group filters of size filter_length:

        # Model Inputs #
        sentences = Input(shape=(self.m, self.n), name='sentences')
        features = Input(shape=[features_size], name='features')
        subj_ids = Input(shape=(None, 1), name='subj_ids')
        obj_ids = Input(shape=(None, 1), name='obj_ids')
        subj_corrp_ids = Input(shape=(None, 1), name='subj_corrp_ids')
        obj_corrp_ids = Input(shape=(None, 1), name='obj_corrp_ids')

        ##############################
        # Define Convolution Encoder #
        ##############################
        cnn = Sequential(name='cnnencoder')
        cnn.add(Convolution1D(input_shape=(self.m, self.n),
                                     nb_filter=1, filter_length=self.convshape, border_mode='valid',
                                     activation='relu', subsample_length=1))

        cnn.add(MaxPooling1D(pool_length=cnn.output_shape[1]))
        cnn.add(Flatten())

        # adding features to encoded vector of CNN
        rel = cnn(sentences)
        rel = Dense(self.sentenceencodedsize)(rel)
        rel = merge([rel, features], mode="concat") # add features to encoded vector from sentences
        rel = Dense(32)(rel)
        rel = Dropout(0.2)(rel)
        rel = Activation('relu')(rel)
        rel = Dense(self.embeddingsize)(rel)
        rel = Reshape([1, 1, self.embeddingsize])(rel)

        self.cnnencoder = Model(input=[sentences, features], output=[rel])

        #####################
        # Entity Embeddings #
        #####################

        ent_emb = Sequential()
        ent_emb.add(Embedding(self.ent_nb, self.embeddingsize, input_length=1))

        sub_emb = ent_emb(subj_ids)
        obj_emb = ent_emb(obj_ids)
        subj_corrp_emb = ent_emb(subj_corrp_ids)
        obj_corrp_emb = ent_emb(obj_corrp_ids)

        sub_emb = Reshape([1, 1, self.embeddingsize])(sub_emb)
        obj_emb = Reshape([1, 1, self.embeddingsize])(obj_emb)

        ##################################
        # DISTMULT: Reconstruction model #
        ##################################
        # Reference : https://arxiv.org/pdf/1411.4072.pdf
        # Optimization function : Maximize the margin between the correct triples and the corrputed triples
        # Corrupted triples are made by either changing subject or changing the object of the triples
        # Objective function :
        # Max {(r * (s' . o') - r * (s . o) + 1 ), 0 }
        ###################################

        td = merge([sub_emb, obj_emb], mode='mul')    # sub and obj true dot product

        cd1 = merge([subj_corrp_emb, obj_emb], mode=lambda x: x[0] * x[1], output_shape=lambda x: x[0]) # corrupt sub dot product
        cd2 = merge([sub_emb, obj_corrp_emb], mode=lambda x: x[0] * x[1], output_shape=lambda x: x[1])  # corrupt obj dot product
        cd = merge([cd1, cd2], mode='concat', concat_axis=1)

        score = merge([rel, td], mode='dot', dot_axes=3)
        score_corrupt = merge([rel, cd], mode=lambda x: T.batched_tensordot(x[0], x[1], axes=(3, 3)), output_shape=lambda x: (x[1][1], x[1][2], 1))

        margin = merge([score_corrupt, score], mode= lambda x: T.maximum(0, x[0] - x[1] + 1.0), output_shape=lambda x: x[1])
        loss = Lambda(lambda x: T.sum(x), name='lossfunction')(margin)

        self.model = Model(input=[sentences, features, subj_ids, obj_ids, subj_corrp_ids, obj_corrp_ids], output=[loss])
        self.model.compile(optimizer='adagrad', loss='mean_squared_error')

    def fit(self, X, y):
        self.model.fit(X, y, nb_epoch=10, batch_size=32)

