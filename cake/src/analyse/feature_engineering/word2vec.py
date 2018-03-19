# coding:utf-8
import os
from gensim.models import word2vec

from cake.src.utils.utils import help_display

class CalcWord2vec:

	def __init__(self, model_path_, word2vec_param=[100,1,5]):
		self.model_path = model_path_
		self.size = word2vec_param[0]
		self.min_count = word2vec_param[1]
		self.window = word2vec_param[2]

	@help_display("Train W2V")
	def train(self, corpus_file_path):
		"""
		Train a new w2v model.
		@ param `corpus_file_path`
		@ param `safe_model` if safe_model is true, the process of training uses update way to refresh model, 
        and this can keep the usage of os's memory safe but slowly. if safe_model is false, the process of training uses the way that load all 
        corpus lines into a sentences list and train them one time.
		"""
		assert os.path.isfile(corpus_file_path), "Please provide correct corpus path."
		model = word2vec.Word2Vec(word2vec.LineSentence(open(corpus_file_path)), size=self.size, window=self.window, min_count=self.min_count)
		model.save(self.model_path)
		print("Producing word2vec model ... okay!")

	@help_display("Update W2V")
	def update(self, corpus_file_path):
		"""
		Update a existing w2v model.
		@ param `model` existing model.
		@ param `corpus_file_path` corpus used to update existing w2v model.
		"""
		assert os.path.isfile(corpus_file_path), "Please provide correct corpus path."
		assert os.path.isfile(self.model_path), "There is not existing w2v model or you provide a incorrect model path."

		model = word2vec.Word2Vec.load(self.model_path)
		f = open(corpus_file_path)
		train_word_count = model.train(word2vec.LineSentence(f), total_examples=len(f.readlines()), epochs = model.iter)
		model.save(self.model_path)
		print("Update word2vec model done, update word num is:%s" % str(train_word_count))
