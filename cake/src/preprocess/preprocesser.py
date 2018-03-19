# coding:utf-8

class Preprocesser(object):
    """
    Interface for preprocessing modules.
    """
    def __init__(self):
        raise NotImplementedError
    def preprocess_corpus(self, corpus):
        raise NotImplementedError

class EnPreprocesser(Preprocesser):
    def __init__(self):
        raise NotImplementedError

class ZhPreprocesser(Preprocesser):
    def __init__(self):
        raise NotImplementedError
