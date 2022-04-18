"""
Description: Creates a Neural Network that trains on twitter hate speech data and is able to classify that content as
either offensive, hate speech, or neither using a Linear and EmbeddingBag layer.
Author: Andrew Bruneel
Date: 2/20/22

Potentially Useful References:
https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
https://towardsdatascience.com/multiclass-text-classification-using-lstm-in-pytorch-eac56baed8df
"""

import classifier
import argparse
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
from sklearn.metrics import classification_report

# Ensure results can be replicated by manually setting random seed.
torch.manual_seed(1)

# Using correct device to improve run time
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

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


class LSTM(nn.Module):
    """Optional: Implement an LSTM."""
    pass


def load_vocab(tweets):
    """Return a dictionary mapping each word to its index in the vocabulary."""
    # Loads all vocabulary from the processed version of the tweets and then
    # creates indices for each word in the word_to_ix dictionary.
    word_to_ix = {}
    for sent in tweets:
        for word in sent.lower().split():
            word_to_ix.setdefault(word, len(word_to_ix))
    return word_to_ix


def main(args):
	# TODO: Load the data using Pandas dataframes, as in classifier.py.
    # Copied the preprocess method from classifier.py to increase the accuracy of tokens
    # and decrease the amount of useless URL tokens in the dictionary, etc.
    def preprocess(text_string):
        """
        Accepts a text string and replaces:
        1) urls with URLHERE
        2) lots of whitespace with one instance
        3) mentions with MENTIONHERE

        This allows us to get standardized counts of urls and mentions
        Without caring about specific people mentioned
        """
        space_pattern = '\s+'
        giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        mention_regex = '@[\w\-]+'
        parsed_text = re.sub(space_pattern, ' ', text_string)
        parsed_text = re.sub(giant_url_regex, '', parsed_text)
        parsed_text = re.sub(mention_regex, '', parsed_text)
        return parsed_text

    # Reading in the pandas dataframe for our tweet data
    df = pd.read_pickle("data/labeled_data.p")
    # Creating a df solely for the tweet column
    tweets = df.tweet
    # Creating a df for the labels
    label = df['class'].astype(int)
    # TODO: Load the vocabulary.
    # Putting all tweets that have been processed from the dataframe into a list
    # that is used later to train and test on (with train_data)
    processed_tweets = []
    for tweet in tweets:
        tweet = preprocess(tweet)
        processed_tweets.append(tweet)
    # Making a list of tuples that contains each processed tweet and its corresponding
    # class label
    train_data = list()
    for index, tweet in enumerate(processed_tweets):
        train_data.append((tweet, label.iloc[index]))
    # Creating a max length variable for our process_batch method
    max_len = max([len(seq.split()) for seq, label in train_data])
    # Loading in vocab
    tok_to_ix = load_vocab(processed_tweets)
    # TODO: Load pre-trained word embeddings, if using them.
    embeds = api.load('glove-twitter-25').vectors # Change to 'word2vec-google-news-300' for word2vec or 'glove-wiki-gigaword-100' for Wikipedia-trained Glove
    emb_dim = embeds.shape[1]
    # TODO: Build the model (either fully-connected or LSTM).
    # Process_batch method used to bring in words for the DataLoader and place them into tensors
    def process_batch(batch):
        x = torch.zeros((len(batch), max_len), dtype=torch.long)
        y = torch.zeros((len(batch)), dtype=torch.long)
        for idx, (text, label) in enumerate(batch):
            seq = [tok_to_ix[tok] for tok in text.lower().split()]
            x[idx, :len(seq)] = torch.Tensor(seq)
            y[idx] = label
        return x.to(device), y.to(device)

    # Establishing number of classes, learning rate, embedding dimensions, batch size
    num_classes = 3
    learning_rate = 0.001
    emb_dim = 25
    batch_size = 16

    # Creating the model, optimizer and loss function
    model = Net(len(tok_to_ix)+1, emb_dim, num_classes, embeds).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Loading data so it is ready to be trained with
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=process_batch)
    # TODO: Train the model for the specified number of epochs, using batches.
    # Using the Adam optimizer to train our model through a set number of epochs and improve loss over time
    n_epochs = 5
    for epoch in range(n_epochs):
        print("\nEpoch:", epoch)
        model.train()
        for x, y in tqdm(train_dataloader):
            pred_y = model(x)
            loss = loss_fn(pred_y, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("Training loss:", loss.item())
    # TODO: Evaluate the model and compare to classifier.py.
    # Evaluating our model using classification report and using the maximum score for our pred_y_test
    with torch.no_grad():
        model.eval()
        x_test, _ = process_batch(train_data)
        pred_y_test = model(x_test)
        report = classification_report(label, pred_y_test.argmax(dim=1))
        print(report)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_file', default='data/labeled_data.p')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for gradient descent.')
    parser.add_argument('--lowercase', action='store_true', help='Whether to make all text lowercase.')
    parser.add_argument('--pretrained', action='store_true', help='Whether to load pre-trained word embeddings.')
    parser.add_argument('--embed_dim', type=int, default=32, help='Default embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Default hidden layer dimension.')
    parser.add_argument('--batch_size', type=int, default=16, help='Default number of examples per minibatch.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--model', default='ff', choices=['ff', 'lstm'])

    args = parser.parse_args()
    main(args)
