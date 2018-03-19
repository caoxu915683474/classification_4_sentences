import random
import datetime
import numpy as np
import keras
from keras.engine.topology import Layer
from keras import initializers
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
import tensorflow.contrib.layers as layers
import tensorflow as tf
from keras import backend as K
from cake.src.utils.utils import get_kwargs
from cake.src.modelzoo.basic_model import BasicModel


class HAN(BasicModel):

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
        self.dropout = get_kwargs(0.5, "dropout", **kwargs)
        self.hidden_unit = get_kwargs(100, "hidden_unit", **kwargs)
        self.attention_unit = get_kwargs(150, "attention_unit", **kwargs)
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

        # Bi-GRU; output = (batch, timestep, hidden_size)
        rnn_output = Bidirectional(
            GRU(self.hidden_unit,
                activation=self.activation,
                kernel_regularizer=l2(self.reg_lambda),
                recurrent_dropout=self.dropout,
                kernel_initializer="random_uniform",
                return_sequences=True)
        )(emb)
        # Attention mechanism
        att_hidden = TimeDistributed(Dense(self.attention_unit, activation="tanh"))(rnn_output)
        alpha = AttLayer(name="alpha")(att_hidden)
        att_out = K.sum(alpha * rnn_output, axis=1)
        dropout = Dropout(self.dropout)(att_out)
        z = Dense(self.n_class, kernel_regularizer=l2(self.reg_lambda))(dropout)

        # Chose the final_layer activation
        if self.classfication_task_type == 1:
            activation = Activation("sigmoid")(z)
            self.y_tilda = tf.identity(activation, name="y_tilda")
            self.loss_layer = tf.reduce_mean(binary_crossentropy(self.y_hat, self.y_tilda), name="loss")
        elif self.classfication_task_type == 0:
            activation = Activation("softmax")(z)
            self.y_tilda = tf.identity(activation, name="y_tilda")
            self.loss_layer = tf.reduce_mean(categorical_crossentropy(self.y_hat, self.y_tilda), name="loss")


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.init((input_shape[-1], 1))
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        ai = K.exp(K.dot(x, self.W))
        weights = ai / K.repeat(K.sum(ai, axis=1), x.shape[-2])
        return weights

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)
