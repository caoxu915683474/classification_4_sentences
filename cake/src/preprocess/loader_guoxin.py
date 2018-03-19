import os
import csv
import glob

import pandas as pd

class ToxicCommentLoader(object):

    def __init__(self, train_path, test_path, output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.output_path = output_path

    def load(self):
        self.load_train()
        self.load_test()

    def load_train(self):
        files = [fn.split("/")[-1] for fn in glob.glob(self.train_path+"*.xlsx")]
        files.remove("guoxin.xlsx")
        files.remove("guoxin_test.xlsx")

        labeled_data = pd.DataFrame(columns = ["comment", "label"])
        labeled_dict = {}
        for i, f in enumerate(files):
            print("Reading and concactenating %s" % f)
            labeled_dict[f] = i
            df = pd.DataFrame(columns = ["comment", "label"])
            data = pd.read_excel(self.train_path + f, header=0)
            df["comment"] = data["content"]
            df["label"] = [i] * len(data["content"])
            labeled_data = labeled_data.append(df, ignore_index=True)
        print(list(labeled_dict.items()))
        try:
            os.stat(self.output_path)
        except:
            os.mkdir(self.output_path)
        labeled_data.to_csv(self.output_path+"train", quoting=csv.QUOTE_NONNUMERIC)

    def load_test(self):
        df = pd.read_excel(self.test_path, header=0)
        print(df)
        new_df = pd.DataFrame(columns = ["comment"])
        to_add = []

        for line in df["content"]:
            n_line = ""
            for i, word in enumerate(str(line)):
                try:
                    word.encode("utf-8")
                    n_line += word
                except UnicodeEncodeError:
                    pass
            to_add.append(n_line) 
            
        new_df["comment"] = to_add
        new_df.to_csv(self.output_path+"test", quoting=csv.QUOTE_NONNUMERIC, index=False)


if __name__ == "__main__":    
    loader = ToxicCommentLoader(
        train_path="/home/cxpc/Documents/nlp/Sentiment_Analysis/text_classification_guoxin/raw_data/",
        test_path="/home/cxpc/Documents/nlp/Sentiment_Analysis/text_classification_guoxin/raw_data/guoxin_test.xlsx",
        output_path="./data/"
    )
    loader.load()
