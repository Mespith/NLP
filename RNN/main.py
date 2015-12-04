import corpus
import RNN
import numpy as np
import CBOW
import SkipGram

# vocabulary_size = 8000
# X_train, Y_train, vocab = corpus.parseRNN("../Words/1-billion-word-language-modeling-benchmark-master/scripts/training-monolingual", 1000, vocabulary_size)
# model = RNN.RNN(len(vocab) + 1)
# #X_train, Y_train, vocab = corpus.parseRNNFile("number_series_1213.txt", 3)
#
# RNN.train_with_sgd(model, X_train, Y_train)
# model.write_to_file("RNNembeddings.txt", vocab)

#word_to_index, contexts, index_to_word = corpus.NewContextParse("../../Words/1-billion-word-language-modeling-benchmark-master/scripts/training-monolingual/news.2007.en.shuffled", 100000)
word_to_index, contexts, index_to_word = corpus.load(100000)
#foo = CBOW.CBOW(100, word_to_index, contexts, index_to_word)
# CBOW.train_with_sgd(foo, 0.5, 0.1)
# foo.writeToFile()

# sg = SkipGram.SkipGram(100, "../Words/1-billion-word-language-modeling-benchmark-master/scripts/training-monolingual/news.2007.en.shuffled")
# SkipGram.train_with_sgd(sg, 0.2, 0.1)
# sg.writeToFile('SkipEmbeddings.txt')

# corpus.parseTimon("../Words/1-billion-word-language-modeling-benchmark-master/scripts/training-monolingual/news.2007.en.shuffled", 100000)