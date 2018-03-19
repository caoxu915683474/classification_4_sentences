import os
import csv

import pandas as pd

class VeggiesLoader(object):

    def __init__(self, train_path, test_path, output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.output_path = output_path

    def load(self):
        self.load_train()
        # self.load_test()

    def load_train(self):

        raw_str = open(self.train_path, "r").readlines()
        new_df = pd.DataFrame(columns = ["comment", "label"])
        
        comment = []
        label = []
        for i, line in enumerate(raw_str):
            new_items = [itm for itm in line.split("|") if itm != r"\\N"]
            comment += new_items
            label += [i]*len(new_items)
        
        new_df["comment"] = comment
        new_df["label"] = label

        print(new_df)
        try:
            os.stat(self.output_path)
        except:
            os.mkdir(self.output_path)
        new_df.to_csv(self.output_path+"train", index=False, quoting=csv.QUOTE_NONNUMERIC)

    def load_test(self):
        pass

if __name__ == "__main__":    
    loader = VeggiesLoader(
        train_path="/home/cxpc/Documents/nlp/Text_Classification/data/data_veggies/raw.txt",
        test_path="",
        output_path="./data3/"
    )
    loader.load()
