"""
Description: HMM with Viterbi decoding for named entity recognition.
Author: Dr. Korpusik, Andrew Bruneel
Reference: Chen & Narasimhan
Date: 6/29/2020, 3/27/2022
"""

import argparse
import numpy as np
import loader

from sklearn.metrics import classification_report


class HMM():
    """
    Hidden Markov Model (HMM) for named entity recognition.
    Two options for decoding: greedy or Viterbi search.
    """

    def __init__(self, dics, decode_type):
        self.num_words = len(dics['word_to_id'])
        self.num_tags = len(dics['tag_to_id'])

        # Initialize all start, emission, and transition probabilities to 1.
        self.initial_prob = np.ones([self.num_tags])
        self.transition_prob = np.ones([self.num_tags, self.num_tags])
        self.emission_prob = np.ones([self.num_tags, self.num_words])
        self.decode_type = decode_type

    def train(self, corpus):
        """
        TODO: Trains a bigram HMM model using MLE estimates.
        Updates self.initial_prob, self.transition_prob, & self.emission_prob.

        The corpus is a list of dictionaries of the form:
        {'str_words': str_words,   # List of string words
        'words': words,            # List of word IDs
        'tags': tags}              # List of tag IDs

        Each dict's lists all have the same length as that instance's sentence.

        Hint: You should see 90% accuracy with greedy and 91% with Viterbi.
        """

        # Initiliazing probability list to 0 to eventually store initial probs based on MLE
        initial_prob_list = np.zeros(self.num_tags)
        transition_list = np.zeros((self.num_tags,self.num_tags))
        emission_list = np.zeros((self.num_tags,self.num_words))
        # Looping through dictionaries within corpus to count tags in the training corpus
        for dictionary_triplet in corpus:
            # Setting an index to reference for building transition_list
            index = 0
            for tag, word in zip(dictionary_triplet['tags'], dictionary_triplet['words']):
                # Updating initial probability counts
                initial_prob_list[tag] += 1
                # Updating transition counts, checking if index is within list bounds
                if index < (len(dictionary_triplet['tags']) - 1):
                    current_word_tag = dictionary_triplet['tags'][index]
                    future_word_tag = dictionary_triplet['tags'][index+1]
                    # Updating transition_list based on current and future word tags
                    transition_list[current_word_tag][future_word_tag] += 1
                    index += 1
                # Updating emission counts using the current word
                emission_list[tag][word] += 1

        # Normalizing to create probabilties from 0 to 1 for self.initial_prob
        initial_tot = sum(initial_prob_list)
        index = 0
        for prob in initial_prob_list:
            initial_prob_list[index] = prob/initial_tot
            index += 1
        self.initial_prob = initial_prob_list

        # Normalizing to create probabilities from 0 to 1 for self.transition_prob
        transition_tot = sum(sum(transition_list))
        row_index = 0
        col_index = 0
        for row in transition_list:
            col_index = 0
            for prob in row:
                transition_list[row_index][col_index] = prob/transition_tot
                col_index += 1
            row_index += 1
        self.transition_prob = transition_list

        # Normalizing to create probabilities from 0 to 1 for self.emission_prob
        emission_tot = sum(sum(emission_list))
        row_index = 0
        for row in emission_list:
            col_index = 0
            for prob in row:
                emission_list[row_index][col_index] = prob/emission_tot
                col_index += 1
            row_index += 1
        self.emission_prob = emission_list


    def greedy_decode(self, sentence):
        """
        TODO: Decode a single sentence in greedy fashion.

        The first step uses initial and emission probabilities per tag.
        Each word after the first uses transition and emission probabilities.

        Return a list of greedily predicted tags.
        """
        tags = []

        # Setting an array of tag probabilities to find the maximum probability from
        tag_probs = np.zeros(self.num_tags)
        index = 0
        tag = ''
        for word in sentence:
            # If we are looking at the initial word in a sentence
            if index == 0:
                for tag in range(len(tag_probs)):
                    # Updating each tag probability based on the initial and emission probabilities
                    tag_probs[tag] = self.initial_prob[tag] * self.emission_prob[tag][word]
                # Adding most recent tag based on highest probability
                tags.append(np.argmax(tag_probs))
                index += 1
            # If we are past the initial word, use the transition and emission probabilities
            else:
                for tag in range(len(tag_probs)):
                    # Updating each tag probability using transition, emission
                    tag_probs[tag] = self.transition_prob[tags[index-1]][tag] * self.emission_prob[tag][word]
                # Adding most recent tag based on highest probability
                tags.append(np.argmax(tag_probs))
                index += 1

        assert len(tags) == len(sentence)
        return tags

    def viterbi_decode(self, sentence):
        """
        Decode a single sentence using the Viterbi algorithm.
        Return a list of tags.
        """
        tags = []

        # TODO (optional)

        assert len(tags) == len(sentence)
        return tags

    def tag(self, sentence):
        """
        Tag a sentence using a trained HMM.
        """
        if self.decode_type == 'viterbi':
            return self.viterbi_decode(sentence)
        else:
            return self.greedy_decode(sentence)


def evaluate(model, test_corpus, dics, args):
    """Predicts test data tags with the trained model, and prints accuracy."""
    y_pred = []
    y_actual = []
    for i, sentence in enumerate(test_corpus):
        tags = model.tag(sentence['words'])
        str_tags = [dics['id_to_tag'][tag] for tag in tags]
        y_pred.extend(tags)
        y_actual.extend(sentence['tags'])

    # Printing our classification report
    print("*** HIDDEN MARKOV MODEL REPORT: ***")
    print(classification_report(y_pred, y_actual))


def main(args):
    # Load the training data.
    train_sentences = loader.load_sentences(args.train_file, args.lower)
    train_corpus, dics = loader.prepare_dataset(train_sentences, mode='train',
                                                lower=args.lower)

    # Train the HMM.
    model = HMM(dics, decode_type=args.decode_type)
    model.train(train_corpus)

    # Load the validation data for testing.
    test_sentences = loader.load_sentences(args.test_file, args.lower)
    test_corpus = loader.prepare_dataset(test_sentences, mode='test',
                                         lower=args.lower,
                                         word_to_id=dics['word_to_id'],
                                         tag_to_id=dics['tag_to_id'])

    # Evaluate the model on the validation data.
    evaluate(model, test_corpus, dics, args)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='data/eng.train')
    parser.add_argument('--test_file', default='data/eng.val')
    parser.add_argument('--lower', action='store_true', help='Whether to make all text lowercase.')
    parser.add_argument('--decode_type', default='greedy', choices=['viterbi', 'greedy'])

    args = parser.parse_args()
    main(args)
