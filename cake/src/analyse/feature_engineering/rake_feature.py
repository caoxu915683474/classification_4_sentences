# coding:utf-8

from cake.src.analyse.feature_engineering.rake import rake
from cake.src.utils.utils import help_display

class CalcRake:

	def __init__(self):
		pass

	@help_display("RAKE analysis")
	def main(self, content, label, stop_word_path, param, type="cate"):
		"""
		The module is used to get the keyword feature of data using RAKE, which can be retrained based on docu.
		@ param `content` String list of texts, which should be segmented string.
		@ param `labels` String/Integer list indicating the category.
		@ param `type` the tfidf score of each document or category, "docu" or "cate", default "cate".
		"""
		assert type in ["docu", "cate"], "Please select which type, docu or cate."

		rake_object = rake.Rake(stop_word_path, *param) # denote that each word has at least 1 char, each phrase has at least 1 word, each keyword happens at least once in the docu.

		keywords = {}
		if type == "docu":
			for c in content:
				keywords_docu = rake_object.run(c)
				keywords[c] = [x[0] for x in keywords_docu]
		else:
			cate_content = {}
			for c, l in zip(content, label):
				if l in cate_content:
					cate_content[l] += " " + c
				else:
					cate_content[l] = c
			for item in cate_content.items():
				keywords_cate = rake_object.run(item[1])
				keywords[item[0]] = [x[0] for x in keywords_cate]
		return keywords


