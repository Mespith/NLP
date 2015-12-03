import numpy as np
import nltk
import itertools
import os

# This script can parse a corpus.
#
# It creates a list of all the unique word, and a dictionary to
# keep track of the word indices.
#
# By setting the window size the script also extracts contexts. The window
# size specifies how many words before and after a word are in a context.

def parseTimon(filename, lines=100000, window=2):
    sentences = []
    line_nr = 0

    contexts = {}

    print "Parsing lines and creating contexts."

    with open(filename) as f:
        for line in f:
            if line_nr >= lines:
                break

            words = nltk.word_tokenize(line.decode('utf-8'))
            sentences.append(words)
            line_nr += 1

            history = []
            counter = 0

            for word in words:
                if not window:
                    continue

                if counter < window * 2 + 1:
                    history.append(word)
                    counter += 1
                else:
                    history = history[1:]
                    history.append(word)

                if counter <= window:
                    continue
                elif counter < window * 2 + 1:
                    i = counter - (window + 1)
                else:
                    i = window

                if history[i] in contexts:
                    contexts[history[i]].append(tuple(history[:i] + history[i+1:]))
                else:
                    contexts[history[i]] = [tuple(history[:i] + history[i+1:])]

    print "Creating vocabulary and mappings."

    word_freq = nltk.FreqDist(itertools.chain(*sentences))
    vocab = word_freq.most_common()
    index_to_word = [x[0] for x in vocab]
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    print "Finished parsing."

    return word_to_index, contexts, index_to_word

unknown_token = "UNKNOWN_TOKEN"

def parseRNNFile(filename, vocabulary_size = 8000):
    sentences = []
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading file: " + filename
    with open(filename, 'rb') as f:
        # This can be removed if every sentence is on a new line
        for line in f:
            # Append SENTENCE_START and SENTENCE_END
            sentences.append(line.decode('utf-8'))
        print "Parsed %d sentences." % (len(sentences))

    # Tokenize the sentences into words
    tokenized_sentences = [sent.split() for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    #print "Using vocabulary size %d." % vocabulary_size
    #print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print "Finished parsing. Creating training data."

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    print "Finished training data."

    return X_train, Y_train, vocab

def parseRNN(directory, nr_of_sentences, vocabulary_size = 8000):
    sentences = []
    for filename in os.listdir(directory):
        # Read the data and append SENTENCE_START and SENTENCE_END tokens
        print "Reading file: " + filename
        with open(directory+'/'+filename, 'rb') as f:
            # This can be removed if every sentence is on a new line
            for line in f:
                # Append SENTENCE_START and SENTENCE_END
                sentences.append(line.decode('utf-8'))
                if len(sentences) >= nr_of_sentences:
                    break
            print "Parsed %d sentences." % (len(sentences))
        if len(sentences) >= nr_of_sentences:
            break

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    voc_size = vocabulary_size
    if voc_size > len(word_freq.items()):
        voc_size = len(word_freq.items())
    print "Found %d unique words tokens." % len(word_freq.items())

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(voc_size)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    print "Using vocabulary size %d." % voc_size
    #print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print "Finished parsing. Creating training data."

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    print "Finished training data."

    return X_train, Y_train, vocab