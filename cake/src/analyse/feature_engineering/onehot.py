# coding:utf-8

import numpy as np
from collections import Counter
from sklearn import tree

from cake.src.utils.utils import help_display

class OneHot:

	def __init__(self):
		pass

	def pre_process(self, content, word):
		corpus = []
		for i, c in enumerate(content):
			seg = c.split(" ")
			c_label = []
			for j, w in enumerate(word):
				c_label.append(1 if w in seg else 0)
			corpus.append(c_label)
		return corpus

	@help_display("OneHot")
	def main(self, content, label, type="bow", k=10000):
		"""
		The module is used to get the keyword feature of data using onehot or tree, which cannot be retrained.
		@ param `content` String list of texts, which should be segmented string.
		@ param `label` String/Integer list indicating the category.
		@ param `type` which method to be used to get the keyword, "bow" or "tree".
		@ param `k` if the type is "cate", top k keywords will be returned, default 1000.
		"""
		assert type in ["bow", "tree"], "Please select which type, docu or cate."
		assert int(k), "Please define appropriate k to indicate how many keywords will be returned."

		words = []
		for item in content:
			words.extend(item.split())

		if type == "bow":
			word_freq = Counter(words)
			return_size = min(k, len(set(words)))
			keywords = []
			for letter, count in word_freq.most_common(return_size):
				keywords.append(letter)
		else:
			words = list(set(words))
			data_set = self.pre_process(content, words)
			clf = tree.DecisionTreeClassifier()
			clf = clf.fit(data_set, label)
			return_size = min(clf.max_features_, k)
			feature_importances_ = dict(zip(range(len(clf.feature_importances_)), clf.feature_importances_))
			feature_importances_ = sorted(feature_importances_.items(), key = lambda x : x[1], reverse=True)
			keywords_index = [x[0] for x in feature_importances_[:return_size]]
			keywords = [words[i] for i in keywords_index]
		return keywords
