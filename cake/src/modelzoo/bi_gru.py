import random
import datetime
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers import InputLayer, Reshape, Input
from keras import optimizers
from keras.regularizers import l2
from keras.losses import categorical_crossentropy, binary_crossentropy
from cake.src.modelzoo.basic_model import BasicModel
import tensorflow as tf

from cake.src.utils.utils import get_kwargs

class Bi_GRU(BasicModel):

    def __init__(self, **kwargs):

        try:
            self.V = kwargs["V"]
            self.seq_len = kwargs["seq_len"]
            self.word_emb_mode = kwargs["word_emb_mode"]
            self.n_class = kwargs["n_class"]
            self.classfication_task_type = kwargs["task_type"]
        except:
            print("V, seq_len, n_class, word_emb_mode are needed for %s" % type(self).__name__)
            exit(1)

        self.word_emb_size = get_kwargs(300, "word_emb_size", **kwargs)
        self.hidden_layers = get_kwargs([100,100], "hidden_layers", **kwargs)
        self.activation = get_kwargs("relu", "activation", **kwargs)
        self.dropout = get_kwargs(0.5, "dropout", **kwargs)
        self.reg_lambda = get_kwargs(0.0, "reg_lambda", **kwargs)

        self.build_model()

    def build_model(self):

        # Model input
        self.define_input()
        model = Sequential()
        if self.word_emb_mode == 1:
            model.add(InputLayer(input_tensor=self.model_input))
            model.add(Embedding(self.V, self.word_emb_size, input_length=self.seq_len))
        else:
            model.add(InputLayer(input_tensor=self.model_input))

        for i, h_layer in enumerate(self.hidden_layers):
            return_seq = True if i != len(self.hidden_layers) - 1 else False
            model.add(Bidirectional(GRU(h_layer,
                                        activation=self.activation,
                                        kernel_regularizer=l2(self.reg_lambda),
                                        recurrent_dropout=self.dropout,
                                        kernel_initializer='random_uniform',
                                        return_sequences=return_seq)))

        # Model output and loss func definition
        if self.classfication_task_type == 1:
            model.add(Dense(self.n_class, activation='sigmoid'))
            self.y_tilda = tf.identity(model.output, name="y_tilda")
            self.loss_layer = tf.reduce_mean(binary_crossentropy(self.y_hat, self.y_tilda), name="loss")
        elif self.classfication_task_type == 0:
            model.add(Dense(self.n_class, activation='softmax'))
            self.y_tilda = tf.identity(model.output, name="y_tilda")
            self.loss_layer = tf.reduce_mean(categorical_crossentropy(self.y_hat, self.y_tilda), name="loss")

        print(self.loss_layer)
