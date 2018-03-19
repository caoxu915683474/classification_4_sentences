# coding:utf-8

from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from cake.src.utils.utils import help_display

class CalcLda:

	def __init__(self):
		pass

	def pre_process(self,content, stop_word_path):
		with open(stop_word_path, "r") as in_file:
			stop = in_file.readlines()
		stop = {}.fromkeys([x.strip() for x in stop])
		text = []
		for line in content:
			line = line.split()
			text.append([x for x in line if x not in stop])
		return text

	def str_to_dic(self, s):
		topic = s[0]
		words = s[1].split("+")
		word_prob = {}
		for i, w in enumerate(words):
			seg = w.split("*")
			word_prob[seg[1].replace("\"","")] = seg[0]
		return topic, word_prob
		
	@help_display("LDA analysis")
	def main(self, content, label, stop_word_path, k):
		"""
		The module is used to get the keyword feature of data using LDA, which cannot be retrained.
		The method not sutable for short text.
		@ param `content` String list of texts, which should be segmented string.
		@ param `label` String/Integer list indicating the category.
		@ param `k` if the type is "cate", top k keywords will be returned, default 50.
		"""
		n_class = len(set(label))
		n_topics = n_class if n_class!=1 else 10

		text = self.pre_process(content, stop_word_path)
		dictionary = Dictionary(text)
		corpus = [dictionary.doc2bow(t) for t in text]
		lda = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=n_topics)

		topics_words_str = lda.print_topics(num_topics=n_topics, num_words=k)
		topics_words_dic = {}

		for item in topics_words_str:
			topic, word_prob = self.str_to_dic(item)
			topics_words_dic[topic] = word_prob.keys()

		return topics_words_dic

