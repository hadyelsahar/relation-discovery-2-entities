import numpy as np
import operator

class Glove:

    def __init__(self, fname, word_index=None):
        """
        :param fname: glove file path
        :param wordidx: replace the glove words with their ids from the give dictionary
        :param max_vocab: if given limit the returned embeddings to the top given ids sorted by order
        """
        self.embedding_size = None
        self.embeddings = dict()
        f = open(fname)
        for line in f:
            values = line.split()
            word = values[0]

            coefs = np.asarray(values[1:], dtype='float32')
            if self.embedding_size is None:
                self.embedding_size = len(coefs)

            self.embeddings[word] = coefs
        f.close()

        # add zeros vector for padding and random for oov
        self.embeddings["__PADDING__"] = np.zeros(self.embedding_size, dtype='float32')
        self.embeddings["__OOV_WORD__"] = np.random.uniform(-1, 1, self.embedding_size)

        embedding_matrix = np.zeros((len(word_index), self.embedding_size))

        # if wordidx are give replace words in keys by the given idx
        for word, i in word_index.items():
            embedding_vector = self.embeddings.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        self.embedding_matrix = embedding_matrix






