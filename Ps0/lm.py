"""
Description: This program creates a language model that can learn on training data and be evaluated. This program makes use of add-alpha smoothing
to increase the accuracy and generalization of the language model.
Date: January 31, 2022
Author: Andrew Bruneel
"""

import math
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk import download
download('punkt')

class LanguageModel:
    """Implements a bigram language model with add-alpha smoothing."""

    def __init__(self, args):
        self.alpha = args.alpha
        self.train_tokens = self.tokenize(args.train_file)
        self.val_tokens = self.tokenize(args.val_file)

        # Use only the specified fraction of training data.
        num_samples = int(args.train_fraction * len(self.train_tokens))
        self.train_tokens = self.train_tokens[: num_samples]
        self.vocab = self.make_vocab(self.train_tokens)
        self.token_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.bigrams = self.compute_bigrams(self.train_tokens)

    def get_indices(self, tokens):
        """Converts each of the string tokens to indices in the vocab."""
        return [self.token_to_idx[token] for token in tokens if token in self.token_to_idx]

    def compute_bigrams(self, tokens):
        """Populates probability values for a 2D np array of all bigrams."""
        counts = np.zeros((len(self.vocab), len(self.vocab)), dtype=float)
        probs = np.zeros((len(self.vocab), len(self.vocab)), dtype=float)
        tokens = self.get_indices(tokens)

        # TODO: Count up all the bigrams.
        # This loop serves to find the counts for each connected bigram within my
        # set of tokens.
        for i in range(len(tokens)-1):
            counts[tokens[i]][tokens[i+1]] += 1

        # TODO: Estimate bigram probabilities using the counts (and alpha).
        # This loop computes probability using add-alpha smoothing as we've discussed
        # in class. The formula for probs[row][col] is directly from what we've seen
        # before and I use bigram_sum to compute the sum of counts for all instances
        # where the first word appears in connection with an arbitrary second word
        # (as per the formula).
        for row in range(len(probs)-1):
            bigram_sum = 0
            bigram_sum = np.sum(counts[row])
            for col in range(len(probs[row])-1):
                probs[row][col] = (counts[row][col] + self.alpha)/(bigram_sum + self.alpha*len(self.vocab))

        return probs

    def compute_perplexity(self, tokens):
        """Evaluates the LM by calculating perplexity on the given tokens."""
        tokens = self.get_indices(tokens)

        # TODO: Sum up all the bigram log probabilities in the test corpus.
        # This loop computes the log probability for my tokens, which is needed
        # to compute the perplexity. I have a case for when i = 0 because otherwise
        # I run into a math domain error, but otherwise I am summing up the
        # probabilities which will be used in later computation.
        log_prob = 0
        N = len(tokens)
        for i in range(N):
            if i == 0:
                log_prob += math.log(np.sum(self.bigrams[tokens[i]]), 2)
            log_prob += math.log(self.bigrams[tokens[i]][tokens[i-1]], 2)

        #for i in range(len(tokens)-1):
                #log_prob += math.log(self.bigrams[tokens[i]][tokens[i+1]],2)
        # This is just the final perplexity formula, which raises the sum of log
        # probabilities to the power of 2 as it is multiplied by N (which is the
        # length of our list of tokens).
        ppl = math.pow(2,(1/N)*log_prob)
        # TODO: Be sure to divide by the number of tokens, not the vocab size!

        return ppl

    def tokenize(self, corpus):
        """Split the given corpus file into tokens on spaces (or with nltk)."""
        # This method simply opens our corpus file and uses word_tokenize() to
        # find tokens. Alternatively, I wrote corpus.split() below as another
        # option in the comments
        f = open(corpus)
        return word_tokenize(f.read())
        # possible : self.tokenized_corpus = corpus.split()

    def make_vocab(self, train_tokens):
        """Create a vocabulary dictionary that maps tokens to frequencies."""
        # This method goes through our train_tokens and updates the frequency
        # of words based on how much they appear in train_tokens. I used
        # vocab_dict.get(word, 0) in case a key-value pair is empty on the first
        # pass-through.
        vocab_dict = dict()
        for word in train_tokens:
            vocab_dict[word] = vocab_dict.get(word, 0) + 1
        return vocab_dict

    def plot_vocab(self, vocab):
        """Plot words from most to least common with frequency on the y-axis."""
        # This method just loops through all vocab words and creates a list of
        # their frequencies based on their values in the vocab dict. I then
        # use reverse sort to order everything properly and plot the graph,
        # noting that Zipf's law holds
        x_vals = []
        y_vals = []
        for word in self.vocab:
            y_vals.append(self.vocab[word])
        y_vals.sort(reverse = True)
        plt.plot(y_vals)
        plt.xlabel('vocab words')
        plt.ylabel('frequency')
        plt.show()


def main(args):
    lm = LanguageModel(args)

    # TODO: implement tokenize(), make_vocab(), and plot_vocab()

    # TODO: Plot the frequency of words by setting command-line arg show_plot.
    if args.show_plot:
        lm.plot_vocab(lm.vocab)

    # TODO: Plot training and validation perplexities as a function of alpha.
    # Hint: Expect ~136 for train and 530 for val when alpha=0.017

    # Was not able to get this running but this is my code attempt
    # Works by creating a list of alpha values needed and computes perplexity
    # adjusting our language model's self.alpha each time running through the
    # loop. Then plots the filled out alpha and perplexity vectors.
    # ppl_vals_one is for training, ppl_vals_two is for validation
    figure, axis = plt.subplots(2, 2)
    alpha_vals = [.00001, .0001, .001, .01, .1, 1, 10]
    ppl_vals_one = []
    ppl_vals_two = []
    for alpha in alpha_vals:
        lm.alpha = alpha
        ppl_vals_one.append(lm.compute_perplexity(lm.train_tokens))
        ppl_vals_two.append(lm.compute_perplexity(lm.val_tokens))
    # training data
    axis[0,0].plot(alpha_vals, ppl_vals_one)
    axis[0,0].xlabel('alpha value')
    axis[0,0].ylabel('training perplexity')
    # validation data
    axis[0,1].plot(alpha_vals, ppl_vals_two)
    axis[0,1].xlabel('alpha value')
    axis[0,1].ylabel('validation perplexity')

    # TODO: Plot train/val perplexities for varying amounts of training data.
    # Similarly to the alpha vs. perplexity plot, I was not able to get this
    # working. This works in mostly the same way except I adjust the argument
    # values through the class vars and then create new language models based
    # on that. I then use .show() at the end to show all finished subplots on
    # the same graph.
    train_percent = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
    ppl_vals_training_one = []
    ppl_vals_training_two = []
    for percent in train_percent:
        args.alpha = .001
        args.train_fraction = percent
        lm = LanguageModel(args)
        ppl_vals_training_one.append(lm.compute_perplexity(lm.train_tokens))
        ppl_vals_training_two.append(lm.compute_perplexity(lm.val_tokens))
    # training data
    axis[1,0].plot(train_percent, ppl_vals_training_one)
    axis[1,0].xlabel('training percent')
    axis[1,0].ylabel('training perplexity')
    # validation data
    axis[1,1].plot(train_percent, ppl_vals_training_two)
    axis[1,1].xlabel('training percent')
    axis[1,1].ylabel('validation perplexity')
    # show all plots
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='lm-data/brown-train.txt')
    parser.add_argument('--val_file', default='lm-data/brown-val.txt')
    parser.add_argument('--train_fraction', type=float, default=1.0, help='Specify a fraction of training data to use to train the language model.')
    parser.add_argument('--alpha', type=float, default=0.0001, help='Parameter for add-alpha smoothing.')
    parser.add_argument('--show_plot', type=bool, default=False, help='Whether to display the word frequency plot.')

    args = parser.parse_args()
    main(args)
