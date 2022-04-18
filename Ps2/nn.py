"""
Description: Implements a neural network for Part-of-Speech tagging
Author: Andrew Bruneel
Date: 3/27/2022
"""

import argparse
import numpy as np
import loader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report


torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Tagger(nn.Module):
    """TODO: Implement a neural network tagger model of your choice."""

    def __init__(self, embed_dim, hidden_dim, vocab_size, num_y):
        super().__init__()
        # Building our Neural Network with an embedding, linear, and softmax layer
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, num_y)
        self.softmax = nn.Softmax(dim = 1)


    def forward(self, text):
        embeds = self.emb(text)
        return self.softmax(self.linear(embeds))


def main(args):

    # Load the training data.
    train_sentences = loader.load_sentences(args.train_file, args.lower)
    train_corpus, dics = loader.prepare_dataset(train_sentences, mode='train', lower=args.lower)
    vocab_size = len(dics['word_to_id'])

    # TODO: Build the model.

    # Setting base hyperparameters
    embed_dim = 25
    # Hidden dimension set to arbitrary value (not used)
    hidden_dim = 25
    vocab_size = len(dics['word_to_id'])
    num_y = len(dics['tag_to_id'])
    # learning rate of .001 found to consistently decrease loss
    learning_rate = .001
    # Building our model and optimizer
    model = Tagger(embed_dim, hidden_dim, vocab_size+1, num_y).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Using a cross entropy loss function because we have multiple classes
    loss_fn = nn.CrossEntropyLoss()
    # TODO: Train the NN model for the specified number of epochs.
    # Training on 10 epochs
    n_epochs = 10
    for epoch in range(n_epochs):
        print("\nEpoch:", epoch)
        model.train()
        for dictionary_triplet in tqdm(train_corpus):
            # Creating lists to store words and tags that will be evaluated through our network
            tag_list = list()
            word_list = list()
            for tag, word in zip(dictionary_triplet['tags'], dictionary_triplet['words']):
                tag_list.append(tag)
                word_list.append(word)
            # Converting words, tags to tensors that can be inputs for our Neural Net
            tensor_word = torch.LongTensor(word_list)
            tensor_tag = torch.LongTensor(tag_list)
            pred_y = model(tensor_word)
            # Running our loss function on the predicted tag vs. actual tag
            loss = loss_fn(pred_y, tensor_tag)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("Training loss:", loss.item())
    # Load the validation data for testing.
    test_sentences = loader.load_sentences(args.test_file, args.lower)
    test_corpus = loader.prepare_dataset(test_sentences, mode='test',
                                         lower=args.lower, word_to_id=dics['word_to_id'],
                                         tag_to_id=dics['tag_to_id'])

    # Building our list of actual/predicted tags to compare later
    test_tags = list()
    pred_tag_list = list()
    # TODO: Evaluate the NN model and compare to the HMM baseline.
    with torch.no_grad():
        for dictionary_triplet in test_corpus:
            # Compiling words so that our neural network can have the same tensor shape passed
            # in as training
            word_compiler = list()
            for word, tag in zip(dictionary_triplet['words'], dictionary_triplet['tags']):
                word_compiler.append(word)
                # Appending our test tags for classification report
                test_tags.append(tag)
            tensor_word = torch.LongTensor(word_compiler)
            pred_y = model(tensor_word)
            # Using argmax with dimension 1 to keep an associated tag for each word
            pred_y_tag = torch.argmax(pred_y, dim = 1)
            # Converting our predicted tags to a list, then extending our current
            # predicting tag list to later be used in classification report
            pred_y_tag = pred_y_tag.tolist()
            pred_tag_list.extend(pred_y_tag)
        report = classification_report(pred_tag_list, test_tags)
        print("*** NEURAL NET REPORT: ***")
        print(report)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='data/eng.train')
    parser.add_argument('--test_file', default='data/eng.val')
    parser.add_argument('--lower', action='store_true', help='Whether to make all text lowercase.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for gradient descent.')
    parser.add_argument('--embed_dim', type=int, default=32, help='Default embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Default hidden layer dimension.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--model', default='lstm', choices=['ff', 'lstm'])

    args = parser.parse_args()
    main(args)
