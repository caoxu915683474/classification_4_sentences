import csv
import numpy as np
import pandas as pd

from cake.src.utils.data_helper import *

class BatchManager(object):

    def __init__(self, **kwargs):

        # self.file_pointer = open(kwargs["input_file"], "r")
        self.train_path, self.dev_path = split_train_dev(kwargs["input_file"])
        self.n_class = kwargs["n_class"]
        self.seq_len = kwargs["seq_len"]
        self.task_type = kwargs["task_type"] # label or classify
        self.batch_size = kwargs["batch_size"]
        self.word_emb_mode = kwargs["word_emb_mode"]
        if self.word_emb_mode == 1:
            self.embed_model, self.inv_vocab = build_vocab(kwargs["word_emb_path"])
        elif self.word_emb_mode == 0:
            self.embed_model = load_word2vec_model(kwargs["word_emb_path"])
        # Get vocabulary size
        self.V = len(self.embed_model)
        self.configs = {**kwargs, **{"V": self.V}}
        
    def iter_on_batch(self, file_to_iter, num_epochs):
        """
        Generates a batch eachtime its acutally needed. 
        Maintain file_pointer, load and embed batch of lines at train-time.
        """
        if file_to_iter == "train":
            file_pointer = open(self.train_path, "r")
        elif file_to_iter == "dev":
            file_pointer = open(self.dev_path, "r")
        raw_sents = ['' for _ in range(self.batch_size)]
        labels = [0 for _ in range(self.batch_size)]
        epoch = 0
        reader = csv.DictReader(file_pointer, delimiter=",")
        while epoch < num_epochs:
            try:
                for i in range(self.batch_size):
                    entry = reader.__next__()
                    label = entry["label"]
                    raw_sents[i] = entry["comment"]
                    labels[i] = np.fromstring(label[1:-1], sep=",") if self.task_type==1 else np.eye(self.n_class)[int(label)] # or classify
                seg_sents = tokenize(raw_sents)
                seg_sents = normalize_sentence(seg_sents, max_length_words=self.seq_len)
                batch_x = embed_sentence(seg_sents, self.embed_model) # default word2vec
                batch_y = labels
                yield (batch_x, batch_y)
            except StopIteration:
                file_pointer.seek(0)
                file_pointer.__next__()
                epoch += 1

    def iter_on_file(self, file_to_iter):
        """
        Generates a batch given a csv file. 
        Maintain file_pointer, load and embed batch of lines at train-time.
        """
        if file_to_iter == "train":
            file_pointer = open(self.train_path, "r")
        elif file_to_iter == "dev":
            file_pointer = open(self.dev_path, "r")
        df = pd.read_csv(file_pointer, header=0, converters={"comment":str})
        X = tokenize(df["comment"].tolist())
        X = normalize_sentence(X, max_length_words=self.seq_len)
        X = embed_sentence(X, self.embed_model)

        if self.task_type == 1:
            Y = [np.fromstring(y[1:-1], sep=",")for y in list(df["label"])]
        else:
            Y = [np.eye(self.n_class)[int(y)] for y in list(df["label"])]

        return X, Y
