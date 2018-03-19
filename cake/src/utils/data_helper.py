"""
This is the definition of utility functions for data preprocessing.
"""
import sys
import codecs
import itertools
import csv

import numpy as np
import pandas as pd
import pickle
from collections import Counter
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split

def build_vocab(word_dict_path):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    print('Building vocab...')
    word_list = [line.strip() for line in open(word_dict_path, "r").readlines()]
    vocabulary_inv = ['<PAD/>','<OOV/>'] + word_list
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def tokenize(sentences):
    """
    Input list of parsed raw training/testing sentences, 
    Returns list of tokenize(segmented) & embedded sentences.
    """
    # Segmentation & Split to list by word
    try:
        sentences = [s.split() for s in sentences]
    except:
        sentences = []
    return sentences

def normalize_sentence(sentences, padding_word='<PAD/>', max_length_words = None):
    """
    Normalizes to max sentence length if max_length_words = None.
    Pads shorter sentences using the padding_word. 
    Crop longer sentences to max_length_words.
    sentences - list of sentence where each sentence is a list of words. eg. [['foo','bar'], ['fo2','bar2']]
    """
    max_length_words = max_length_words if max_length_words is not None\
                                        else max(len(x) for x in sentences)
    norm_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = max_length_words - len(sentence)
        padded_sentence = sentence + [padding_word] * num_padding
        chopped_sentence = padded_sentence[:max_length_words]
        norm_sentences.append(chopped_sentence)
    return norm_sentences

def embed_sentence(sentences, embed_model):
    """
    Input list of padded/segmented training/testing sentences, 
    and embedding dictionary {word2vec(0) or 1-of-N(1)}.
    Returns list of embedded word vectors.
    """
    if type(list(embed_model.values())[0]) == int: # 1-of-N
        sentences = [[embed_model.get(word, 1) for word in sentence] for sentence in sentences]
    else: # word2vec
        default_embedding = [0.0]*len(list(embed_model.values())[0])
        sentences = [[embed_model.get(word, default_embedding) for word in sentence] for sentence in sentences]
    return sentences

def load_word2vec_model(dict_name):
    """
    Load word2vec_model txt file to embedding dictionary.
    """
    print("building embedding dictionary..")
    with open(dict_name, 'r') as f:
        f.readline() #Neglect Header
        embed_model = {line.split()[0]: np.fromstring(' '.join(line.split()[1:]),\
                        sep=' ', dtype=float)  for line in f.readlines()}
    return embed_model

def shuffle_data(sentences, labels):
    shuffle_indices = np.random.permutation(np.arange(len(sentences)))
    sentences = sentences[shuffle_indices]
    labels = labels[shuffle_indices]
    return sentences, labels

def split_data_4_CV(data, K, k, ratio = 0.05): 
    """
    Input preprocessed data, split for K-fold cross validation.
    Ouput splitted train, dev, test set.
    """
    SPLIT = len(data)/K
    CUR_FOLD = k
    train = np.concatenate((data[:CUR_FOLD*SPLIT], data[(CUR_FOLD+1)*SPLIT:]), axis=0)
    test = data[CUR_FOLD*SPLIT:(CUR_FOLD+1)*SPLIT]

    # train, dev = train_test_split(train, test_size=ratio)
    train, dev = train[:-1*int(len(data)*ratio)], train[-1*int(len(data)*ratio):]
    return train, dev, test

def split_train_dev(input_file, ratio=0.12):

    df = pd.read_csv(input_file)
    df = df.sample(frac=1).reset_index(drop=True)
    train, dev = df.iloc[:-1*int(len(df)*ratio)], df.iloc[-1*int(len(df)*ratio):]
    print("Splitted to train: %i; dev: %i" % (len(train), len(dev)))

    train_path = "%s/train.csv" % "/".join(input_file.split("/")[:-1])
    dev_path = "%s/dev.csv" % "/".join(input_file.split("/")[:-1])
    train.to_csv(train_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    dev.to_csv(dev_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    return train_path, dev_path

def data_helper(raw_sents, word_emb_mode, word_emb_path, seq_len):
    """
    Wrap the data preparation into a function to be called conveniently.
    """
    if word_emb_mode == 1:
        embed_model, inv_vocab = build_vocab(word_emb_path)
    elif word_emb_mode == 0:
        embed_model = load_word2vec_model(word_emb_path)
    seg_sents = tokenize(raw_sents)
    seg_sents = normalize_sentence(seg_sents, max_length_words=seq_len)
    processed_sents = embed_sentence(seg_sents, embed_model) # default word2vec
    return processed_sents

