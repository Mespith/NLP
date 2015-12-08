import numpy as np
from numpy.random import random_sample
from scipy.stats import rv_discrete
import time
from scipy.special import expit
import sys

class CBOW:

    # EmbeddingDim is the number of dimensions you want to embed the words in,
    # filenameIn and filenameOut are strings of the file containing the corpus
    # and the file you want to write the word vectors to
    def __init__(self, embeddingDim, nr_of_samples, word_to_index, contexts, index_to_word, word_counts):
        self.N = embeddingDim
        self.K = nr_of_samples
        self.word_to_index = word_to_index
        self.contexts = contexts
        self.wordlist = index_to_word
        self.wordcount = word_counts
        self.V = len(self.word_to_index)

        # input weights
        self.W_i = np.random.normal(0, 0.1, (self.V, self.N))

        # output weights
        self.W_o = np.random.normal(0, 0.1, (self.V, self.N))

        power_wordcount = np.power(self.wordcount.values(), (3./4.))
        Z = np.sum(power_wordcount)/ np.sum(self.wordcount.values())**(3./4.)
        self.distrib = rv_discrete(values=(range(self.V), power_wordcount/Z))  # This defines a Scipy probability distribution


    # X is a vector with the indices corresponding to 1 in the one hot
    # encoding for the context input words
    def weighted_values(self, values, probabilities, size):
        bins = np.add.accumulate(probabilities)
        return values[np.digitize(random_sample(size), bins)]

    def computeHiddenLayer(self, X):
        h = np.zeros(self.W_i.shape[1])
        for i in X:
            h += self.W_i[i]
        return h/len(X)

    # Performs stochastic training descent to train the weights
    def sgd_step(self, learning_rate):

        indices = np.arange(self.V)
        np.random.shuffle(indices)
        self.Error = 0
        for i in indices:
            try:
                contexts_i = self.contexts[i]
            except KeyError:
                contexts_i = {}
            for context in contexts_i:
                X = [word for word in context]
                h = self.computeHiddenLayer(X)
                #K = self.weighted_values(self.p_values, np.arange(self.V), self.K)
                v_samplewords = np.array(self.distrib.rvs(size=self.K))
                sig = expit(self.W_o[i].dot(h)) - 1
                self.W_o[i] -= learning_rate * sig * h
                for x in X:
                    self.W_i[x] -= 1/len(X) * learning_rate * sig * self.W_o[i]
                for l in v_samplewords:
                    if (i == l):
                        t = 1
                    else:
                        t = 0
                    sig = expit(np.dot(self.W_o[l], h))-t
                    self.W_o[l] -= learning_rate * sig * h
                    for x in X:
                        self.W_i[x] -= 1/len(X) * learning_rate * sig * self.W_o[l]
                for l in v_samplewords:
                    temp = self.W_o[l].dot(h)
                    if temp < 0:
                        self.Error += np.log(np.exp(temp)+1)-temp
                    else:
                        self.Error += np.log(1 + np.exp(-temp))

                temp = self.W_i[i].dot(h)
                if temp < 0:
                    self.Error += np.log(np.exp(temp)+1)-temp
                else:
                    self.Error += np.log(1 + np.exp(-temp))

    def sgd_update(self, dW_o, dW_i):
        self.W_o += dW_o
        self.W_i += dW_i

    def writeToFile(self, outfile1, outfile2, vocabfile):
        with open(vocabfile, 'w') as vocfile:
            with open(outfile1, 'w') as fout:
                for i, banana in enumerate(self.W_o):
                    banana = ' '.join([str(bite) for bite in banana])
                    vocfile.write(self.wordlist[i] + '\n')
                    fout.write(banana + '\n')

        with open(outfile2, 'w') as fout:
            for i, banana in enumerate(self.W_i):
                banana = ' '.join([str(bite) for bite in banana])
                fout.write(banana + '\n')

# performs stochastic training descent to train the weights
def train_with_sgd(model, learning_rate, tolerance, min_lr):
    errors = []
    min_error = sys.maxint
    best_Wo = model.W_o
    best_Wi = model.W_i
    model.Error = sys.maxint
    oldError = 0
    while (tolerance < abs(oldError-model.Error) and learning_rate > min_lr):
        print time.ctime() + " >> Error: " + str(model.Error)
        errors.append(model.Error)
        if (len(errors) > 1 and errors[-1] > errors[-2]):
                learning_rate *= 0.8
                print "Setting learning rate to %f" % learning_rate
        elif model.Error < min_error:
            best_Wi = model.W_i
            best_Wo = model.W_o
            min_error = model.Error
            print "Found new best model!"
        oldError = model.Error
        model.sgd_step(learning_rate)

    model.W_o = best_Wo
    model.W_i = best_Wi