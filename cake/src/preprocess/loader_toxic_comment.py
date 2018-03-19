import os
import csv

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
        df = pd.read_csv(self.train_path)
        new_df = pd.DataFrame(columns = ["comment", "label"])
        
        def f_label(x):
            if 1 in list(x):
                return list(x) + [0]
            else:
                return list(x) + [1]

        def f_neutral(x):
            if 1 in list(x):
                return 1
            else:
                return 0

        new_df["comment"] = df["comment_text"]
        new_df["label"] = df.loc[:,"toxic":"identity_hate"].apply(f_label, axis=1)
        new_df["neutral"] = df.loc[:,"toxic":"identity_hate"].apply(f_neutral, axis=1)
        # print(new_df["neutral"].value_counts())

        try:
            os.stat(self.output_path)
        except:
            os.mkdir(self.output_path)
        new_df.to_csv(self.output_path+"train", quoting=csv.QUOTE_NONNUMERIC)

    def load_test(self):
        df = pd.read_csv(self.test_path)
        new_df = pd.DataFrame(columns = ["comment"])
        new_df["comment"] = df["comment_text"]
        new_df.to_csv(self.output_path+"test", quoting=csv.QUOTE_NONNUMERIC, index=False)

if __name__ == "__main__":    
    loader = ToxicCommentLoader(
        train_path="/home/cxpc/Documents/nlp/Text_Classification/data/data_toxic-comment/train.csv",
        test_path="/home/cxpc/Documents/nlp/Text_Classification/data/data_toxic-comment/test.csv",
        output_path="./data2/"
    )
    loader.load()
