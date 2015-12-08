import corpus
import RNN
import numpy as np
import CBOWNS as CBOW
import SkipGram

# vocabulary_size = 8000
# X_train, Y_train, vocab = corpus.parseRNN("../Words/1-billion-word-language-modeling-benchmark-master/scripts/training-monolingual", 1000, vocabulary_size)
# model = RNN.RNN(len(vocab) + 1)
# #X_train, Y_train, vocab = corpus.parseRNNFile("number_series_1213.txt", 3)
#
# RNN.train_with_sgd(model, X_train, Y_train)
# model.write_to_file("RNNembeddings.txt", vocab)

# word_to_index, contexts, index_to_word, index_count = corpus.NewContextParse("../../Words/1-billion-word-language-modeling-benchmark-master/scripts/training-monolingual/news.2007.en.shuffled", True)
word_to_index, contexts, index_to_word, word_count = corpus.load(500000)
foo = CBOW.CBOW(100, 3, word_to_index, contexts, index_to_word, word_count)
CBOW.train_with_sgd(foo, 0.5, 1., 0.01)
foo.writeToFile("CBOW_Wout.txt", "CBOW_Win.txt", "CBOW_labels.txt")

# sg = SkipGram.SkipGram(100, "../Words/1-billion-word-language-modeling-benchmark-master/scripts/training-monolingual/news.2007.en.shuffled")
# SkipGram.train_with_sgd(sg, 0.2, 0.1)
# sg.writeToFile('SkipEmbeddings.txt')

# corpus.parseTimon("../Words/1-billion-word-language-modeling-benchmark-master/scripts/training-monolingual/news.2007.en.shuffled", 100000)