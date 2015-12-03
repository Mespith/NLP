import numpy as np
import datetime
import sys
from scipy.special import expit

class RNN:
    def __init__(self, vocab_dim, hidden_dim=100, bptt_truncate=5):
        # Assign the variable values
        self.vocab_dim = vocab_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly instantiate the parameters
        self.U = np.random.uniform(-np.sqrt(1./vocab_dim), np.sqrt(1./vocab_dim), (hidden_dim, vocab_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (vocab_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, s):
        z = self.V.dot(s)
        #norm = np.sum(np.exp(z))
        #return np.sum(np.exp(z)/norm)
        max = np.max(z)
        e = np.exp(z - max)
        dist = e / np.sum(e)

        return dist

    def forward_propagation(self, x):
        # The total number of timesteps
        T = len(x)
        # Save the hidden layer values in S
        # The initial hidden layer is 0
        S = np.zeros((T+1, self.hidden_dim))
        S[-1] = np.zeros(self.hidden_dim)
        # Save the outputs of each timestep in O
        O = np.zeros((T, self.vocab_dim))

        for t in range(T):
            S[t] = expit(self.U[:,x[t]] + self.W.dot(S[t-1]))
            O[t] = self.softmax(S[t])

        return [O, S]

    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0.
        # Walk over all the sentences
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # Only select the outputs of the correct words.
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # The correct prediction is 1, therefore we increase our loss with the difference between 1 and the calculated probability.
            correct_word_predictions = np.ma.log(correct_word_predictions)
            L += np.sum(correct_word_predictions)
        return L

    def calculate_loss(self, x, y):
        # Normalize the total loss.
        N = np.sum((len(y_i) for y_i in y))
        return -1 * (self.calculate_total_loss(x, y)/N)

    def bptt(self, x, y):
        T = len(y)
        o, s = self.forward_propagation(x)
        # Initialize the gradients
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.

        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))

            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta_t
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)

        return [dLdU, dLdV, dLdW]

    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Update the parameters
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    def write_to_file(self, filename, vocab):
        with open(filename, 'w') as fout:
            for i, banana in enumerate(self.U):
                banana = ' '.join([str(bite) for bite in banana])
                fout.write(vocab[i][0].encode('utf-8') + ' ' + banana + '\n')

def train_with_sgd(model, X_train, y_train, learning_rate=0.1, nepoch=100, evaluate_loss_after=5):
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust learning learning rate
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # Loop through the training samples
        for i in range(len(y_train)):
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1