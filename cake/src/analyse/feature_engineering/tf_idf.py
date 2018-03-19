# coding:utf-8

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from cake.src.utils.utils import help_display

class CalcTfidf:

	def __init__(self):
		self.vectorizer = CountVectorizer()
		self.transformer = TfidfTransformer()

	def get_tfidf(self, corpus):
		tfidf = self.transformer.fit_transform(self.vectorizer.fit_transform(corpus))
		word = self.vectorizer.get_feature_names()
		weight = tfidf.toarray()
		return word, weight

	def get_weight_docu(self, corpus, word, weight, k):
		weight_corpus = {}
		for i, c in enumerate(corpus):
			seg = c.split(" ")
			weight_vec = [weight[i][word.index(s)] if s in word else 0 for s in seg]
			weight_corpus[c] = [str(x) for x in weight_vec]
		return weight_corpus

	def get_weight_cate(self, cates, word, weight, k):
		weight_cates = {}
		unique_cates = set(cates)
		for i, cate in enumerate(unique_cates):
			dic = dict(zip(word, weight[i]))
			dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
			return_size = min(k, len(dic))
			weight_cates[cate] = [x[0] for x in dic][:return_size]
		return weight_cates

	@help_display("TFIDF analysis")
	def main(self, content, label, type="cate", k=10):
		"""
		The module is used to get the keyword feature of data using Tf-Idf, which cannot be retrained.
		@ param `content` String list of texts, which should be segmented string.
		@ param `label` String/Integer list indicating the category.
		@ param `type` the tfidf score of each document or category, "docu" or "cate", default "cate".
		@ param `k` if the type is "cate", top k keywords will be returned, default 10.
		"""
		assert type in ["docu", "cate"], "Please select which type, docu or cate."
		assert int(k), "Please define appropriate k to indicate how many keywords will be returned."

		if type == "docu":
			word, weight = self.get_tfidf(content)
			return self.get_weight_docu(content, word, weight, k)
		else:
			cate_content = {}
			for c, l in zip(content, label):
				if l in cate_content:
					cate_content[l] += " " + c
				else:
					cate_content[l] = c
			corpus = list(cate_content.values())
			word, weight = self.get_tfidf(corpus)
			return self.get_weight_cate(label, word, weight, k)
