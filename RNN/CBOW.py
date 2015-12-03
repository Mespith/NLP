import numpy as np
import corpus
import time

class CBOW:

    # EmbeddingDim is the number of dimensions you want to embed the words in,
    # filenameIn and filenameOut are strings of the file containing the corpus
    # and the file you want to write the word vectors to
    def __init__(self, EmbeddingDim, filenameIn, outfile1, outfile2):
        self.N = EmbeddingDim
        self.vocab, self.contexts, self.wordlist = corpus.parseTimon(filenameIn)
        self.outfile1 = outfile1
        self.outfile2 = outfile2
        self.V = len(self.vocab)

        # input weights
        self.W_i = np.random.normal(0, 0.1, (self.N, self.V))

        # output weights
        self.W_o = np.random.normal(0, 0.1, (self.V, self.N))


    # X is a vector with the indices corresponding to 1 in the one hot
    # encoding for the context input words
    def computeHiddenLayer(self, X):
        h = np.zeros(self.W_i.shape[0])
        for i in X:
            h += self.W_i[:, i]
        return h/len(X)


    # dots is the vector Wh, j corresponds to the index with 1 for the correct output word
    def computeError(self, dots, biggest, j):
        return dots[j]-biggest-np.log(np.sum(np.exp(dots-biggest)))


    def computeSoftMax(self, dots, biggest):
        return np.exp(dots-biggest)/np.sum(np.exp(dots-biggest))


    # index is the index of the correct output word
    def updateWeights(self, h, X, index, dots, biggest, eta):
        y = self.computeSoftMax(dots, biggest)
        y[index] = y[index] - 1
        for c in X:
            self.W_i[:, c] -= eta/len(X)*np.dot(self.W_i, y)
        self.W_o -= eta*np.outer(y, h)


    def writeToFile(self):
        with open(self.outfile1, 'w') as fout:
            for i, banana in enumerate(self.W_o):
                banana = ' '.join([str(bite) for bite in banana])
                fout.write(self.wordlist[i].encode('utf-8') + ' ' + banana + '\n')

        with open(self.outfile2, 'w') as fout:
            for i, banana in enumerate(self.W_i.T):
                banana = ' '.join([str(bite) for bite in banana])
                fout.write(self.wordlist[i].encode('utf-8') + ' ' + banana + '\n')

# performs stochastic training descent to train the weights
def train_with_sgd(model, learningRate, tolerance):
    errors = []
    model.Error = tolerance + 1
    oldError = 0
    while (tolerance < abs(oldError-model.Error)):
        print time.ctime() + " >> Error: " + str(model.Error)
        errors.append(model.Error)
        if (len(errors) > 1 and errors[-1] < errors[-2]):
                learningRate = learningRate * 0.5
                print "Setting learning rate to %f" % learningRate

        indices = np.arange(model.V)
        np.random.shuffle(indices)
        oldError = model.Error
        model.Error = 0
        for i in indices:
            word_i = model.wordlist[i]
            try:
                contexts_i = model.contexts[word_i]
            except KeyError:
                contexts_i = []
            for context in contexts_i:
                X = [model.vocab[word] for word in context]
                h = model.computeHiddenLayer(X)
                dots = np.dot(model.W_o, h)
                biggest = np.max(dots)
                model.updateWeights(h, X, i, dots, biggest, learningRate)
                model.Error += model.computeError(dots, biggest, i)