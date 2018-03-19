# coding: utf-8

import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import groupby

from cake.src.utils.utils import help_display

class BasicAnalyse:

    def __init__(self):
        pass

    def get_cate_length(self, corpus, label_type=0):
        """
        Count the text length of each category.
        @ param `label_type` indicate classification or multilabelling, 0 or 1.
        """
        if label_type == 0:
            corpus_grouped = corpus["comment"].groupby(corpus["label"])
            cate_length = {item[0]:[len(x.split()) for x in list(item[1])] for item in corpus_grouped}
        else:
            pattern = "\d"
            num_labels = len(re.findall(pattern, list(corpus["label"])[0]))
            cate_length = {}.fromkeys(range(num_labels))
            indexs = list(corpus.index)
            for idx in indexs:
                text = len(corpus.ix[idx,"comment"].split())
                cates = re.findall(pattern, corpus.ix[idx,"label"])
                for idy in range(num_labels):
                    if int(cates[idy]) == 1:
                        if cate_length[idy]:
                            cate_length[idy].append(text)
                        else:
                            cate_length[idy] = [text]
        return cate_length

    # 画饼图
    def plot_pie(self, dic, title, output_folder):
        fig = plt.figure(figsize=(10,10))
        plt.pie(dic.values(), labels = dic.keys(), autopct="%1.2f%%", radius=3)
        plt.title(title)
        plt.savefig("%s/%s.png" %(output_folder, title))


    # 画柱状图
    def plot_hist(self, dic, output_folder):
        f, ax = plt.subplots(len(dic), sharex=False, figsize=(20,6*len(dic)))
        statistic = []
        for i, item in enumerate(dic.items()):
            counter = Counter(item[1]).most_common()
            ax[i].set_title("%s document length distribution" % item[0])
            ax[i].hist(item[1], bins=len(counter), color="Orange")
            ax[i].set_xlim(0, max(item[1]))
            statistic.append([item[0], max(item[1]), min(item[1])])
        plt.savefig("%s/length_versus_cates_hitsplot.png" % output_folder)
        plt.close()
        return statistic

    def balance_data(self, corpus, cate_distribute):
        print("-------------")
        multiple = {}
        max_cate = max(cate_distribute.values())
        for item in cate_distribute.items():
            multiple[item[0]] = max_cate/cate_distribute[item[0]]
        grouped_corpus = corpus["comment"].groupby(corpus["label"])
        balanced_content = []
        balanced_label = []
        for item in grouped_corpus:
            balanced_content.extend(item[1] * multiple[item[0]])
            balanced_label.extend([item[0]] * len(item[1]) * multiple[item[0]])
        balanced = pd.DataFrame(columns=["comment", "label"])
        balanced["content"] = balanced_content
        balanced["label"] = balanced_label
        return balanced

    # 基础统计分析: 类别分布，长度分布，各类别长度分布，balance or not
    def run_statistic_analysis(self, corpus, balance_or_not=False, label_type=0):
        cate_length = self.get_cate_length(corpus, label_type)
        cate_distribute = {item[0]:len(item[1]) for item in cate_length.items()}
        if balance_or_not:
            corpus = pd.DataFrame(columns = ["comment", "label"])
            corpus = self.balance_data(corpus, cate_distribute)
        return cate_length, cate_distribute, corpus

    @help_display("BasicAnalysis")
    def analyse(self, input_file, output_folder, balance_or_not=False, label_type=0):
        print("Loading data...")
        corpus_df = pd.read_csv(input_file, header=0, converters={"comment":str})

        print("Running analysis...")
        cate_length, cate_distribute, corpus = self.run_statistic_analysis(corpus_df, balance_or_not, label_type)

        print("Generating plot and report...")
        self.plot_pie(cate_distribute, "Category distribution", output_folder)
        statistics = self.plot_hist(cate_length, output_folder)
        # write the report
        with open("%sanalyse_report.txt" % output_folder, "w+") as out_file:
            out_file.write("----Category distribution----\n")
            for item in cate_distribute.items():
                out_file.write("%s:%s\n" % (str(item[0]), str(item[1])))
            out_file.write("----Statistic of each category----\n")
            for item in statistics:
                out_file.write("%s:%s,%s\n" % (str(item[0]), str(item[1]), str(item[2])))

