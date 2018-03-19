import random
import datetime
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Flatten
from keras.layers import Dropout, Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers import *
from keras import optimizers
from keras import regularizers
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.regularizers import l2
import tensorflow as tf
from keras import backend as K

from cake.src.utils.utils import get_kwargs
from cake.src.modelzoo.basic_model import BasicModel

class Word_CNN(BasicModel):

    def __init__(self, **kwargs):
        
        try:
            self.V = kwargs["V"]
            self.seq_len = kwargs["seq_len"]
            self.n_class = kwargs["n_class"]
            self.word_emb_mode = kwargs["word_emb_mode"]
            self.classfication_task_type = kwargs["task_type"]
        except:
            print("V, seq_len, n_class, word_emb_mode are needed for %s" % type(self).__name__)
            exit(1)

        self.word_emb_size = get_kwargs(300, "word_emb_size", **kwargs)
        self.activation = get_kwargs("relu", "activation", **kwargs)
        self.filter_sizes = get_kwargs([3,4,5], "filter_sizes", **kwargs)
        self.num_filters = get_kwargs(100, "num_filters", **kwargs)
        self.dropout = get_kwargs(0.5, "dropout", **kwargs)
        self.hidden_unit = get_kwargs(100, "hidden_unit", **kwargs)
        self.reg_lambda = get_kwargs(0.0, "reg_lambda", **kwargs)

        self.build_model()

    def build_model(self):
        
        # Define Input layer & Tensor
        self.define_input()

        # Embedding
        if self.word_emb_mode == 1:
            emb = Embedding(
                    self.V,
                    self.word_emb_size,
                    input_length=self.seq_len,
                    name="embedding"
                )(self.model_input)
        else:
            emb = self.model_input
        # Conv layers
        conv_blocks = []
        for sz in self.filter_sizes:
            conv = Conv1D(filters=self.num_filters,
                          kernel_size=sz,
                          padding="valid",
                          activation=self.activation,
                          strides=1)(emb)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
            
        # Concatenate multi-granularity conv results
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        z = Dropout(self.dropout)(z)
        z = Dense(self.hidden_unit, activation=self.activation)(z)
        z = Dense(
            self.n_class,
            kernel_regularizer=l2(self.reg_lambda),
        )(z)

        # Chose the final_layer activation
        if self.classfication_task_type == 1:
            activation = Activation("sigmoid")(z)
            self.y_tilda = tf.identity(activation, name="y_tilda")
            self.loss_layer = tf.reduce_mean(binary_crossentropy(self.y_hat, self.y_tilda), name="loss")
        elif self.classfication_task_type == 0:
            activation = Activation("softmax")(z)
            self.y_tilda = tf.identity(activation, name="y_tilda")
            self.loss_layer = tf.reduce_mean(categorical_crossentropy(self.y_hat, self.y_tilda), name="loss")

        print(self.loss_layer)