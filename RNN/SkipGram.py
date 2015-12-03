import numpy as np
import corpus
import time

class SkipGram:
    def __init__(self, N, filename):
        self.N = N
        self.vocab, self.contexts, self.wordlist = corpus.parseTimon(filename, 100)
        self.V = len(self.vocab)
        self.W1 = np.ones((N, self.V)) * 0.5
        self.W2 = np.ones((self.V, N)) * 0.5

    def updateWeights(self, EI, h, i, eta): #EI is as in (31) word2vec parameter learning explained, i is the index of the input word
        EH = np.dot(self.W2.T, EI) #EH is as in (36)
        self.W1[:,i] -= eta*EH
        self.W2 -= eta*np.outer(EI, h)


    def forward(self, index, learningRate): #index is the index corresponding to 1 in the one hot encoded input word
        h = self.W1[:, index]
        u = np.dot(self.W2, h)
        biggest = np.max(u)
        y = np.exp(u - biggest)/np.sum(np.exp(u - biggest)) #compute the softmax of u
        try:
            Contexts = self.contexts[self.wordlist[index]] #get the contexts associated with word
        except KeyError:
            Contexts = []
        E = 0 #The error from the word hot encoded with index
        for context in Contexts: #for each context compute the error
            EI = y * len(context)
            E = 0
            for word in context:
                i = self.vocab[word]
                EI[i] -= 1
                E -= u[i]
            self.updateWeights(EI, h, index, learningRate)
            E += len(context)*(biggest + np.log(np.sum(np.exp(u - biggest))))
        return E

    def writeToFile(self, filename):
        with open(filename, 'w') as fout:
            for i, banana in enumerate(self.W1):
                banana = ' '.join([str(bite) for bite in banana])
                fout.write(self.wordlist[i].encode('utf-8') + ' ' + banana + '\n')

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
        for i in indices:
            model.Error += model.forward(i, learningRate)