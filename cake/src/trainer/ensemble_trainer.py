# coding:utf-8

import os
import sys
import numpy as np
import tensorflow as tf
import keras as K 

class Ensemble_Trainer(object):
    
    def __init__(self, model_paths, raw_data, real_label, output_folder):
        """
        Constructer.
        @ param `model_paths` list of model paths.
        @ param `input_data` data used to train the stacked model.
        @ param `output_folder` where to save the stacked model.
        """
        self._model_paths = model_paths
        self.raw_data = raw_data
        self.real_label = real_label
        self.output_folder = output_folder

    def data_helper(self, result_type="label"):
        pass

    def train(self):
        pass
