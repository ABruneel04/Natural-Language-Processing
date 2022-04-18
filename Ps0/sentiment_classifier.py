"""
Description: This program tests varying classifiers on training and validation sets to compute their accuracies. We
look through a Multinomial Bayesian Classifier as well as Logistic Regression models using unigrams and bigrams. These
can all be compared against our base case which simply returns 1 as a classification and scores an accuracy of ~50%
Date: January 31, 2022
Author: Andrew Bruneel
"""

import argparse
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


class BaselineClassifier():
    """This baseline classifier always predicts positive sentiment."""

    def __init__(self, args):
        self.train_sents, self.train_labels = self.read_data(args.train_file)
        self.val_sents, self.val_labels = self.read_data(args.val_file)

    def read_data(self, filename):
        """Extracts all the sentences and labels from the input file."""
        sents = []
        labels = []
        with open(filename) as f:
            for line in f.readlines():
                line = line.strip().split()
                sents.append(line[1:])
                labels.append(int(line[0]))
        return sents, labels

    def predict(self, corpus):
        """Always predicts a value of 1 given the input corpus."""
        # Simply returns 1 as class label prediction rather than making
        # an educated guess
        return 1

    def evaluate(self):
        """Evaluates accuracy on training and validation predictions."""
        # Runs through the training and validation sets and assigns a label
        # for each sentence by calling predict(). Once this is done, totals
        # up the number of correct label assignments and divides by the total
        # number of sentences in each set to find a fraction of how many are
        # correct. Returns a tuple order (training correct, validation correct)
        train_num_correct = 0
        train_percent = 0
        val_num_correct = 0
        val_percent = 0
        for sent in self.train_sents:
            if(self.predict(sent) == self.train_labels[self.train_sents.index(sent)]):
                train_num_correct += 1
        train_percent = train_num_correct/(len(self.train_labels))
        for sent in self.val_sents:
            if(self.predict(sent) == self.val_labels[self.val_sents.index(sent)]):
                val_num_correct += 1
        val_percent = val_num_correct/(len(self.val_labels))
        return(train_percent, val_percent)

class NaiveBayesClassifier(BaselineClassifier):
    """Implements Naive Bayes with unigram features using sklearn."""

    def __init__(self, args):
        super().__init__(args)
        self.token_to_idx = self.extract_unigrams()
        # TODO: Assign a new MultinomialNB() to self.classifier.
        self.classifier = MultinomialNB()
        self.train()

    def extract_unigrams(self):
        """Builds a dictionary of unigrams mapping to indices."""
        # TODO: For each training sentence, assign each new token to an index.
        # This method builds our token index dictionary we will use to identify
        # tokens that will be used as features later on. We simply loop through
        # our training data and identify tokens based on if words already have an
        # index value within our dictionary
        token_idx_dict = dict()
        idx_counter = 0
        for sent in self.train_sents:
            for token in sent:
                # Finding each token within our training sentences
                if token not in token_idx_dict:
                    # If the token is not inside our dictionary, we give it an index value
                    token_idx_dict[token] = idx_counter
                    idx_counter += 1
        return token_idx_dict

    def compute_features(self, sents):
        """"Convert sents to np array of feature vectors X."""
        X = np.zeros((len(sents), len(self.token_to_idx)), dtype=float)
        for index, feat_vec in enumerate(tqdm(X, desc='Load unigram feats')):
            for token in sents[index]:
                token_idx = self.token_to_idx[token]
                X[index][token_idx] += 1
        return X

    def train(self):
        """Trains a Naive Bayes classifier on given input x and labels y."""
        # This method computes features based on the tokens from our training
        # sentences and then trains our MultinomialNB classifier based on
        # the given training labels and features we have acquired. It then
        # fits the MultinomialNB based on those two inputs.
        # TODO: Compute features X from self.train_sents.
        # vectorizer = CountVectorizer()
        # vectorizer.fit_transform(self.train_sents)
        X = self.compute_features(self.train_sents)
        # TODO: Convert train_labels to a numpy array of labels y.
        y = np.array(self.train_labels)
        # TODO: Fit the classifier on X and y.
        self.classifier.fit(X, y)
        # Able to see 90+ accuracy on the training set for MultinomialNB
        # as well as for LogisticRegression
        print(self.classifier.score(X, y))

    def predict(self, corpus):
        """Makes predictions with the classifier on computed features."""
        # This method simply uses the .predict() method of our MultinomialNB
        # and assigns predicted labels to all of our validation sentences.
        # We use compute_features to create the 2D array needed to run through
        # self.classifier.predict() and then score ourselves to find accuracy
        # Was not able to get this model to adjust the fit for validation data
        vectorizer = CountVectorizer()
        vectorizer.fit_transform(open(args.val_file))
        X = self.val_sents
        y = self.classifier.predict(self.compute_features(X))
        score = self.classifier.score(X, y)
        return score


class LogisticRegressionClassifier(NaiveBayesClassifier):

    """Implements logistic regression with unigram features using sklearn."""
    def __init__(self, args):
        BaselineClassifier.__init__(self, args)
        self.token_to_idx = self.extract_unigrams()
        # TODO: Assign a new LogisticRegression() to self.classifier.
        self.classifier = LogisticRegression()
        # Hint: You can adjust penalty and C params with command-line args.
        self.train()


class BigramLogisticRegressionClassifier(LogisticRegressionClassifier):
    """Implements logistic regression with unigram and bigram features."""

    def __init__(self, args):
        BaselineClassifier.__init__(self, args)
        self.token_to_idx = self.extract_unigrams()
        self.bigrams_to_idx = self.extract_bigrams()
        # TODO: Assign a new LogisticRegression() to self.classifier.
        self.classifier = LogisticRegression()
        # Hint: Be sure to set args.solver.
        self.train()

    def extract_bigrams(self):
        """Builds a dictionary of bigrams mapping to indices."""
        # This method is very similar to our unigram extractor we made before
        # but instead stores bigrams within the dictionary as tuples. This allows
        # us to use the same method of assigning index via counter and move through
        # our training sentences to assign bigrams in a similar way to tokens
        bigram_idx_dict = dict()
        idx_counter = 0
        for sent in self.train_sents:
            for index in range(len(sent)-1):
                # Finding each bigram within our training sentences
                bigram = (token[index], token[index+1])
                if bigram not in bigram_idx_dict:
                    # If the bigram is not inside our dictionary, we give it an index value
                    bigram_idx_dict[bigram] = idx_counter
                    idx_counter += 1
        return bigram_idx_dict

    def compute_features(self, sents):
        """Convert sents to np array of feature vectors X."""
        # TODO: Include both unigram and bigram features.
        # For this method I simply adapted what we did earlier for computing features.
        # X_one corresponds to our unigrams, and X_two corresponds to bigrams.
        # The method then returns a feature vector for each parameter so that
        # We can score them separately using our classifier and compare their
        # accuracies.
        X_one = np.zeros((len(sents), len(self.token_to_idx)), dtype=float)
        for index, feat_vec in enumerate(tqdm(X_one, desc='Load unigram feats')):
            for token in sents[index]:
                token_idx = self.token_to_idx[token]
                X_one[index][token_idx] += 1
        X_two = np.zeros((len(sents), len(self.bigram_to_idx)), dtype=float)
        for index, feat_vec in enumerate(tqdm(X_two, desc='Load bigram feats')):
            for token in sents[index]:
                token_idx = self.bigram_to_idx[token]
                X_two[index][token_idx] += 1
        return (X_one, X_two)



def main(args):
    # TODO: Evaluate basline classifier (i.e., always predicts positive).
    # Hint: Should see roughtly 50% accuracy.
    b = BaselineClassifier(args)
    print(b.evaluate())
    # TODO: Evaluate Naive Bayes classifier with unigram features.
    # Hint: Should see over 90% training and 70% testing accuracy.
    nb = NaiveBayesClassifier(args)
    # print(nb.predict(nb.val_sents))
    # TODO: Evaluate logistic regression classifier with unigrams.
    lr = LogisticRegressionClassifier(args)
    # TODO: Evaluate logistic regression classifier with unigrams + bigrams.
    lr_bigram = BigramLogisticRegressionClassifier(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='sentiment-data/train.txt')
    parser.add_argument('--val_file', default='sentiment-data/val.txt')
    parser.add_argument('--solver', default='liblinear', help='Optimization algorithm.')
    parser.add_argument('--penalty', default='l2', help='Regularization for logistic regression.')
    parser.add_argument('--C', type=float, default=1.0, help='Inverse of regularization strength for logistic regression.')

    args = parser.parse_args()
    main(args)
