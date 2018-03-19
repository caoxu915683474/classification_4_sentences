# coding:utf-8

import os
import pandas as pd
from cake.src.analyse.feature_engineering.lda import CalcLda
from cake.src.analyse.feature_engineering.rake_feature import CalcRake
from cake.src.analyse.feature_engineering.tf_idf import CalcTfidf
from cake.src.analyse.feature_engineering.word2vec import CalcWord2vec
from cake.src.analyse.feature_engineering.textrank import CalcTextrank
from cake.src.analyse.feature_engineering.onehot import OneHot
from cake.src.utils.utils import help_display

# Wrapper for catching kwarg not found error.
def get_kwargs(default, *keys, **kwargs):
	try:
		r = kwargs
		for k in keys:
			r = r[k]
		return r
	except (KeyError, TypeError):
		return default

class FeatureEngineer:
	def __init__(self, **kwargs):

		# LDA params
		self.use_lda = get_kwargs(0, "lda", "use_lda", **kwargs)
		# TFIDF params
		self.use_tf_idf = get_kwargs(0, "tf_idf", "use_tf_idf", **kwargs)
		self.tfidf_type = get_kwargs(0, "tf_idf", "type", **kwargs)
		# RAKE params
		self.use_rake = get_kwargs(0, "rake", "use_rake", **kwargs)
		self.rake_type = get_kwargs(0, "rake", "type", **kwargs)
		self.rake_param = get_kwargs([1, 1, 1], "rake", "rake_param", **kwargs)
		# W2V params
		self.use_word2vec = get_kwargs(0, "word2vec", "use_word2vec", **kwargs)
		self.w2v_model_path = get_kwargs(0, "word2vec", "output_path", **kwargs)
		self.word2vec_param = get_kwargs([100, 1, 5], "word2vec", "word2vec_param", **kwargs)
		# TextRank params
		self.use_textrank = get_kwargs(0, "textrank", "use_textrank", **kwargs)
		self.textrank_window = get_kwargs(0, "textrank", "textrank_window", **kwargs)
		# OneHot params
		self.use_onehot = get_kwargs(1, "onehot", "use_onehot", **kwargs)
		self.onehot_type = get_kwargs("bov", "onehot", "type", **kwargs)
		self.onehot_topK = get_kwargs(10000, "onehot", "topK", **kwargs)
		# Stopword path
		self.stop_word_path = get_kwargs("", "stop_word_path", **kwargs)

	@help_display("FeatureEngineer")
	def run(self, corpus_path, report_path, k=10):
		assert os.path.isfile(corpus_path), "Data file not exist."

		print("Loading preprocessed corpus...")
		corpus = pd.read_csv(corpus_path, header=0, converters={"comment":str})
		content = list(corpus["comment"])
		label = list(corpus["label"])

		def write_dic(f, dic):
			for item in dic.items():
				f.write("%s:%s\n" %(str(item[0]), "，".join(list(item[1]))))

		# LDA
		if self.use_lda:
			lda_obj = CalcLda()
			topic_word = lda_obj.main(content, label, self.stop_word_path, k=k)
			with open("%s_lda_result.txt" % report_path, "w+") as out_file:
				out_file.write("----------LDA result -- keywords for each topic----------\n")
				write_dic(out_file, topic_word)

		if self.use_tf_idf:
			tfidf_obj = CalcTfidf()
			tfidf_result = tfidf_obj.main(content, label, type=self.tfidf_type, k=k)
			with open("%s_tfidf_result.txt" % report_path, "w+") as out_file:
				out_file.write("----------TF-IDF result for %s ----------\n" % self.tfidf_type)
				write_dic(out_file, tfidf_result)

		if self.use_rake:
			rake_obj = CalcRake()
			rake_result = rake_obj.main(content, label, self.stop_word_path, self.rake_param,
																			type=self.rake_type)
			with open("%s_rake_result.txt" % report_path, "w+") as out_file:
				out_file.write("----------RAKE result for %s ----------\n" % self.rake_type)
				write_dic(out_file, rake_result)

		if self.use_textrank:
			textrank_obj = CalcTextrank()
			textrank_result = textrank_obj.main(content, content, window=self.textrank_window, k=k)
			with open("%s_textrank_result.txt" % report_path, "w+") as out_file:
				out_file.write("----------TextRank result ----------\n")
				write_dic(out_file, textrank_result)

		if self.use_word2vec:
			w2v = CalcWord2vec(self.w2v_model_path, word2vec_param = self.word2vec_param)
			if os.path.isfile(self.w2v_model_path):
				w2v.update(corpus_path)
			else:
				w2v.train(corpus_path)

		if self.use_onehot:
			onehot_obj = OneHot()
			onehot_result = onehot_obj.main(content, label, type=self.onehot_type, 
																		k=self.onehot_topK)
			with open("%s_onehot_result.txt" % report_path, "w+") as out_file:
				for item in onehot_result:
					out_file.write(item + "\n")
