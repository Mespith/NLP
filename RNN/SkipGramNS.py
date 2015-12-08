import corpus
import numpy as np

# TODO fix calling corpus parser to get actual word frequencies

dim = 100
K = 3
eta = 0.1
epsilon = 0.5
V = 0

# Sigmoid
def sigmoid(vector_or_scalar):
    return 1.0 / (1.0 + np.exp(-vector_or_scalar))


# This should return K samples from a unigram dist.
def getSamples(K):
    return list(np.random.randint(V, size=3))


if __name__ == '__main__':
    word_to_index, contexts, index_to_word = corpus.parse('story.txt')
    V = len(index_to_word)
    Wi = np.ones((V, dim))
    Wo = np.ones((dim, V))

    # Rewrite the contexts as one long list of context words.
    for key in contexts:
        new_context = []

        for context in contexts[key]:
            new_context.extend(context)

        contexts[key] = new_context

    # Set new error just to get into the loop.
    E_old = 0
    E_new = 2*epsilon

    while abs(E_new - E_old) > epsilon:
        E_old = E_new
        E_new = 0

        for i in range(V):
            h = Wi[i]

            for outword in contexts[index_to_word[i]]:
                samplewords = getSamples(K)
                samplewords.append(word_to_index[outword])

                for word in samplewords:
                    if word in contexts[index_to_word[i]]:
                        t = 1
                    else:
                        t = 0

                    vw = Wo[:,word]
                    vw -= eta * (sigmoid(np.dot(vw, h) - t) * h)
                    h -= eta * (sigmoid(np.dot(vw, h) - t) * vw)

                for word in samplewords:
                    E_new -= np.log(sigmoid(np.dot(Wo[:,word], h)))

        print E_new