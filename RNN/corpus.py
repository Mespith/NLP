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
punktuation = ['"', ',', '!', '?', '.', ':', ';', '(', ')', '-', '+', '$', '#', '\'', '@', '%', '&', '*', '[', ']', '\\', '/', '`', '<', '>', '']

# Filename:     The directory of the file you want to parse.
# whole_file:   When true, the parsing doesn't stop after the set number of lines but will go through the whole file.
# lines:        The number of lines you want to parse from the given file.
# window:       The size of the window to create the contexts.
def NewContextParse(filename, whole_file=False, lines=100000, window=2):

    print "Parsing %s lines." % (lines)

    sentences = get_lines(filename, lines, whole_file)

    print "Creating vocabulary and mappings."

    word_freq = nltk.FreqDist(itertools.chain(*sentences))
    vocab = sub_sample(word_freq, 0.000001)
    index_to_word = [x for x in vocab]
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    print "Found %s unique words that survived the sub-sampling." %(len(vocab))
    print "Creating contexts."

    if window:
        contexts = get_contexts(sentences, word_to_index, window)
    else:
        contexts = {}

    print "Created %s contexts." %(len(contexts))
    print "Saving parsed data."

    store(vocab, contexts, len(sentences))

    return word_to_index, contexts, index_to_word, vocab

# lines:    The number of lines define which parsed file you want to load.
def load(lines):
    file_prefix = "parsed_files/" + str(lines)
    file_name = file_prefix + "_vocab.txt"
    vocab = {}

    print "Reading file: " + file_name

    with open(file_name) as f:
        for line in f:
            data = line.split()
            vocab[data[0]] = int(data[1])

    index_to_word = vocab.keys()
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    file_name = file_prefix + "_contexts.txt"
    contexts = {}

    print "Parsed %s words." %(len(vocab))
    print "Reading file: " + file_name

    with open(file_name) as f:
        for line in f:
            data = line.split()
            key = int(data[0])
            values = data[1].split(',')
            if key in contexts:
                contexts[key].append(tuple(map(int, values)))
            else:
                contexts[key] = [tuple(map(int, values))]

    print "Parsed contexts of %s words." %(len(contexts))

    return word_to_index, contexts, index_to_word, vocab

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

def sub_sample(word_freq, t):
    vocab = {}
    for word in word_freq:
        # Sub sampling infrequent words
        if word_freq[word] < 10:
            continue
        # Sub sampling frequent words
        P = 1 - np.sqrt((t / word_freq.freq(word)))
        if np.random.random_sample() > P:
            vocab[word] = word_freq[word]
    return vocab

def get_lines(filename, lines, whole_file=False):
    line_nr = 0
    sentences = []
    # Getting the text in the right format.
    with open(filename) as f:
        for line in f:
            if not whole_file and line_nr >= lines:
                break
            line_nr += 1

            line = line.lower()
            words = nltk.word_tokenize(line.decode('utf-8'))
            sentence = []

            for word in words:
                word = word.strip('",!?.:;()\'').lstrip("'").rstrip("'")
                if word.endswith("'s"):
                    word = word[:-2]
                if word in punktuation:
                    continue
                else:
                    sentence.append(word)

            sentences.append(sentence)
    return sentences

def get_contexts(sentences, word_to_index, window):
    contexts = {}
    # Creating contexts
    for sentence in sentences:
        history = []
        counter = 0
        for word in sentence:
            # Only add a word to a context if it is in the vocabulary.
            if word in word_to_index:
                word_indx = word_to_index[word]
                if counter < window * 2 + 1:
                    history.append(word_indx)
                    counter += 1
                else:
                    history = history[1:]
                    history.append(word_indx)

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
    return contexts

def store(vocab, contexts, lines):
    file_prefix = "parsed_files/" + str(lines)
    file_name = file_prefix + "_vocab.txt"
    with open(file_name, 'w+') as v:
        for word_freq in vocab:
            v.write(word_freq.encode('utf-8') + " " + str(vocab[word_freq]) + "\n")

    file_name = file_prefix + "_contexts.txt"
    with open(file_name, 'w+') as v:
        for word in contexts:
            for context in contexts[word]:
                line = ""
                for cont_word in context:
                    line += str(cont_word) + ","
                v.write(str(word) + " " + line[:-1] + "\n")