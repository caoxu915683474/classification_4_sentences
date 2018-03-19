# coding:utf-8

from textrank4zh import TextRank4Keyword

class CalcTextrank:

	def __init__(self):
		pass

	def main(self, content, label, window=2, k=20):
		"""
		The function is used to exract keywords using TextRank.
		@ param `content` String list of texts, which should be segmented string.
		@ param `label` String/Integer list indicating the category.
		@ param `window` count window.
		@ param `k` top k keywords will be returned, default 10.
		"""
		keywords = {}
		cate_content = {}
		for c, l in zip(content, label):
			if l in cate_content:
				cate_content[l] += " " + c.replace(" ", "")
			else:
				cate_content[l] = c.replace(" ", "")
		for item in cate_content.items():
			tr4w = TextRank4Keyword()
			tr4w.analyze(text=item[1], lower=True, window=window)
			kws = tr4w.get_keywords(k, word_min_len=1)
			keywords[item[0]] = [x.word for x in kws]
		
		return keywords
