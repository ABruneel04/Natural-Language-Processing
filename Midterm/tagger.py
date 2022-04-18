"""
Description: This program creates a Logistic Regression classifier as well as a Neural Network and evaluates how each of these performs
text classification on a given data file
Author: Andrew Bruneel
Date: 3/12/22
"""

import numpy as np
import pandas as pd
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from tqdm import tqdm
import gensim.downloader as api

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

# Ensure results can be replicated by manually setting random seed.
torch.manual_seed(1)

# Using correct device to improve run time
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

class LogRegClassifier():
    """Implements Logistic Regression with unigram features using sklearn."""

    def __init__(self, train_corpus, test_corpus):
        super().__init__()
        # Loads the data into two sets: 80% of which is training, and 20% of which is validation
        self.train_words, self.validation_words, self.train_labels, self.validation_labels = train_test_split(train_corpus, test_corpus, test_size=0.20, random_state=42)
        # Utilizes a CountVectorizer to transform around our training words
        self.vectorizer = CountVectorizer(ngram_range=(1, 1)) # unigrams
        self.vectorizer.fit_transform(self.train_words)
        # Builds Logistic Regression Classifier, increasing the iterations to allow the solution algorithm to converge
        self.classifier = LogisticRegression(max_iter = 200)
        # Training our classifier using our train() method
        self.train()

    def train(self):
        X = self.vectorizer.transform(self.train_words)
        # Convert train_labels to a numpy array of labels y.
        y = np.array(self.train_labels)
        # Fitting the classifier to best apply to our training data
        self.classifier.fit(X, y)

    def predict(self, corpus):
        # Using our LR classifier to predict labels for words given a passed in corpus
        X = self.vectorizer.transform(corpus)
        return self.classifier.predict(X)

    def evaluate(self, eval_words, eval_tags):
        # Using classification report to analyze our LR classifier's performance
        predicted_tags = model.predict(eval_words)
        report = classification_report(eval_tags, predicted_tags)
        return report

class Net(nn.Module):
    """TODO: Implement your own fully-connected neural network!"""
    def __init__(self, num_words, emb_dim, num_y, embeds=None):
        super().__init__()
        # Use embedding bag to improve accuracy (embedding dimensions aligns with pre-trained embeddings)
        self.emb = nn.EmbeddingBag(num_words, emb_dim)
        if embeds is not None:
            self.emb.weight = nn.Parameter(torch.Tensor(embeds))
        # Linear layer to keep a simple Neural Net that learns quickly
        self.linear = nn.Linear(emb_dim, num_y)

    def forward(self, text):
        embeds = self.emb(text)
        return self.linear(embeds)

if __name__=="__main__":
    # Loading in the training data and isolating the words and tags
    df = pd.read_csv("data/trainDataWithPOS.csv")
    words = df['Word']
    tags = df['Tag']
    # Loading in the testing data and isolating the words and tags
    test_df = pd.read_csv("data/testDatawithPOS.csv")
    test_words = df['Word']
    test_tags = df['Tag']
    # Creating a Logistic Regression model with the training data, which will then train on that data
    model = LogRegClassifier(words, tags)
    # Testing our model on the held-out testing data, and printing the classifcation report
    print("*** LOGISTIC REGRESSION REPORT: ***")
    print(model.evaluate(test_words, test_tags))

    def load_vocab(words):
        """Return a dictionary mapping each word to its index in the vocabulary."""
        # Loads all vocabulary from the word file and creates a dictionary to store them
        word_to_ix = {}
        for word in words:
            word_to_ix.setdefault(word, len(word_to_ix))
        return word_to_ix

    # Creating a list of unique tags to index later
    tag_list = pd.unique(tags).tolist()

    # Using a test_size of 0.1 to train on (almost) all data and save the held-out testing set for evaluation later
    # Test size was previously used for validation
    train_words, validation_words, train_labels, validation_labels = train_test_split(words, tags, test_size=0.01, random_state=42)
    # Creating lists for our test words and labels
    nn_test_words = list()
    nn_test_labels = list()
    for index, word in enumerate(test_words):
        nn_test_words.append(word)
        nn_test_labels.append(test_tags.iloc[index])
    # Loading vocabulary for the neural net from all training words
    tok_to_ix = load_vocab(words)

    # Setting num classes to 5 for each possible classification of words in the corpus
    num_classes = 5
    # Parameter adjustments made here
    learning_rate = 0.001
    emb_dim = 25

    # Creating our model with the Adam optimizer and Cross Entropy Loss
    model = Net(len(tok_to_ix)+1, emb_dim, num_classes, embeds=None).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Five epochs for useful learning while keeping efficiency
    n_epochs = 5
    for epoch in range(n_epochs):
        print("\nEpoch:", epoch)
        model.train()
        # Looping through training words and their associated labels
        for word, label in zip(train_words, train_labels):
            # Turning our word into a Tensor of its associated index in our dictionary
            tok = torch.LongTensor([[tok_to_ix[word]]])
            # Also turning our label into a Tensor of its value from tag_list
            label_val = torch.LongTensor([tag_list.index(label)])
            pred_tag = model(tok)
            # Running loss function on its predicted label and associated correct label
            loss = loss_fn(pred_tag, label_val)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("Training loss:", loss.item())

    with torch.no_grad():
        model.eval()
        # Creating a list of predicted tags and actual tags to be used in classification report
        pred_tag_list = list()
        actual_tag_list = list()
        # Looping through testing words and their associated labels
        for word, label in zip(nn_test_words, nn_test_labels):
            tok = torch.LongTensor([[tok_to_ix[word]]])
            pred_tag_test = model(tok)
            # Using argmax to find the most likely label for each word
            index = torch.argmax(pred_tag_test)
            # Converting our integer to the associated string in tag list
            associated_str = tag_list[index]
            # Appending our predicted tag list and actual tag list for the report
            pred_tag_list.append(associated_str)
            actual_tag_list.append(label)
        report = classification_report(pred_tag_list, actual_tag_list)
        print("*** NEURAL NET REPORT: ***")
        print(report)
