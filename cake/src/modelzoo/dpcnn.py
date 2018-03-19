#coding:utf-8

from keras.layers import Conv1D,MaxPool1D,Dense,Dropout,Input,Flatten,Activation
from keras.layers.embeddings import Embedding
from keras.layers.merge import add
from keras import optimizers
import tensorflow as tf
import math
from cake.src.utils.utils import get_kwargs
from keras.losses import categorical_crossentropy,binary_crossentropy
from keras.regularizers import l2
import numpy as np
from cake.src.modelzoo.basic_model import BasicModel

class DPCNN(BasicModel):
    """
    This is an implement of ACL2017 DPCNN for text classification
    you can use this model to predict the class of each sentences ,and the model
    is long-term friendly

    :param Vocab_len: Length of your vocabulary which should be typed int.
    :param seq_len: Length of your sentence , make sure the length can obtain the most of 
                    your data set sentences.
    :param n_class: The num of classes you are facing in your problem.
    :param word_emb_mode: The type of the embedding mode , 0 for one hot and 1 for word2vec 
                          or other pre-trained vector mode.
    :param classfication_task_type: The task type you are facing :
           #Set 0 for multi-binary task, the final layer loss will be binary_crossentropy 
           and the final
           #layer activation will be sigmoid activation.
           #Set 1 for categorical task, the final layer loss will be categorical_crossentropy 
           and the final
           #layer activation will be softmax activation.

    :param word_emb_size: The embeddingdim of each word ,default is 300
    :param block_num: The block is a unit which contains two Conv1D layer and one MaxPooling1D 
                      layer .Note: the num of block is Restricted make sure your seq_len can be 
                      divided by 2 at (block_num+1) times
    :param conv1D_activation: The activation of conv1D ,default is 'relu'.
    :param dropout: Dropout of your model , default is 0.5.
    :param num_filters: Filters num of each conv1D ,default is 250.
    :param kernel_size: Kernel_size of conv1D , default is 3.
    :param pooling_size: Pooling_size of maxpooling1D ,default is 3.
    :param reg_lambda: Kernel_regularizer of final dense layer ,default is 0.0
    """

    def calcu_block_num(self, seq_len):
        
        return math.floor(math.log2(seq_len)) - 3

    def __init__(self, **kwargs):

        try:
            self.V = kwargs["V"]
            self.seq_len = kwargs["seq_len"]
            self.n_class = kwargs["n_class"]
            self.word_emb_mode = kwargs["word_emb_mode"]
            self.classfication_task_type = kwargs["task_type"]
        except:
            print("V, seq_len, n_class, word_emb_mode, classfication_task_type are need for \
            %s" %type(self).__name__)
            exit(1)

        self.word_emb_size = get_kwargs(300, "word_emb_size", **kwargs)
        self.block_num = get_kwargs(self.calcu_block_num(self.seq_len), "word_emb_size", **kwargs)
        self.conv1D_activation = get_kwargs("relu", "conv1D_activation", **kwargs)
        self.dropout = get_kwargs(0.5, "dropout", **kwargs)
        self.num_filters = get_kwargs(250, "num_filters", **kwargs)
        self.kernel_size = get_kwargs(3, "kernel_size", **kwargs)
        self.pooling_size = get_kwargs(3, "pooling", **kwargs)
        self.reg_lambda = get_kwargs(0.0, "reg_lambda", **kwargs)
        self.build_model()

    def conv_block(self, model):
        """
        The block of dpcnn , which contains one MaxPooling1D Layer and two Conv1D Layer
        We use shortcut and pre-activation in this block
        """
        
        model1 = MaxPool1D(pool_size=self.pooling_size, strides=2)(model)
        model2 = Activation(self.conv1D_activation)(model1)
        model3 = Conv1D(self.num_filters,
                        kernel_size=self.kernel_size,
                        strides=1,
                        padding="same")(model2)
        model4 = Activation(self.conv1D_activation)(model3)
        model5 = Conv1D(self.num_filters,
                        kernel_size=self.kernel_size,
                        strides=1,
                        padding="same")(model4)
        model6 = add([model1, model5])

        return model6

    def build_model(self):
        """
        This method is used to build your own keras model
        This part define the structure of dpcnn
        """

        assert self.seq_len // (2**(self.block_num+1)) > 2, "\nYou should reduce the block_num to make sure the sentence \
         length can be divided by 2 at ( block_num + 1 ) times.\n"

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

        trans_dim = Dense(self.num_filters)(emb)
        pre_activation_layer_0_1 = Activation(self.conv1D_activation)(trans_dim)
        conv_0_1 = Conv1D(self.num_filters,
                          kernel_size=self.kernel_size,
                          strides=1,
                          padding="same")(pre_activation_layer_0_1)
        pre_activation_layer_0_2 = Activation(self.conv1D_activation)(conv_0_1)
        conv_0_2 = Conv1D(self.num_filters,
                          kernel_size=self.kernel_size,
                          strides=1,
                          padding="same")(pre_activation_layer_0_2)
        add_layer_0 = add([trans_dim, conv_0_2])
        # Block repeat
        for i in range(self.block_num):
            add_layer_0 = self.conv_block(add_layer_0)
        final_maxpooling = MaxPool1D(pool_size=self.pooling_size, strides=1)(add_layer_0)
        dropout_layer = Dropout(self.dropout)(final_maxpooling)
        flatten = Flatten()(dropout_layer)
        dense = Dense(self.n_class, kernel_regularizer=l2(self.reg_lambda))(flatten)

        # Chose the final_layer activation
        if self.classfication_task_type == 1:
            activation = Activation("sigmoid")(dense)
            self.y_tilda = tf.identity(activation, name="y_tilda")
            self.loss_layer = tf.reduce_mean(binary_crossentropy(self.y_hat, self.y_tilda), name="loss")
        elif self.classfication_task_type == 0:
            activation = Activation("softmax")(dense)
            self.y_tilda = tf.identity(activation, name="y_tilda")
            self.loss_layer = tf.reduce_mean(categorical_crossentropy(self.y_hat, self.y_tilda), name="loss")

        print(self.loss_layer)
