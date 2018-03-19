import tensorflow as tf
import keras
from keras.layers import InputLayer, Reshape, Input, Concatenate

class BasicModel(object):

    def __init__(self):
        pass

    def define_input(self):
        in_shape = (None, self.seq_len,) if self.word_emb_mode == 1\
                    else (None, self.seq_len, self.word_emb_size)
        x_d_type = tf.int32 if self.word_emb_mode == 1 else tf.float32
        self.x = tf.placeholder(x_d_type, shape=in_shape, name="x")
        self.y_hat = tf.placeholder(tf.float32, shape=(None, self.n_class), name="y_hat")
        self.model_input = Input(tensor=self.x, shape=in_shape)
