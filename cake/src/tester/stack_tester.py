# coding:utf-8

import numpy as np
import sys

from sklearn.externals import joblib
from collections import Counter

from cake.src.tester.ensemble_tester import Ensemble_Tester

def average_stacking(input_data, model_path):
    n_of_clfs = len(input_data)
    compound_probs = np.sum(input_data, axis=0)/n_of_clfs
    pred_y = np.argmax(compound_probs)
    return pred_y

def vote_stacking(input_data, model_path):
    pred_y = []
    for i, item in enumerate(input_data):
        c = Counter(item)
        pred_y.append(c.most_common(1)[0][0])
    return pred_y

def logistic_stacking(input_data, model_path):
    try:
        clf = joblib.load(model_path)
    except Exception as err:
        print("Catch error: %s model not exist, please check the model path to make sure it is correct." % model_path)
        exit(1)
    pred_y = clf.predict(input_data)
    return pred_y

def linear_stacking(input_data, model_path):
    try:
        clf = joblib.load(model_path)
    except:
        print("Catch error: %s model not exist, please check the model path to make sure it is correct." % model_path)
        exit(1)
    pred_y = clf.predict(input_data)
    return pred_y

class Stack_Tester(Ensemble_Tester):
    
    def test(self, input_data):
        """
        @ param `input_data` single piece of data
        @ return predict result of the stacking model
        """
        predict_result = [tester.test(input_data) for tester in self.ensemble_models]
        data = np.array(predict_result).T.tolist() if self.result_type=="label" else predict_result
        dic = {"input_data":data, "model_path":self.clf_path}
        return getattr(sys.modules[__name__], "%s_stacking" % self.method)(**dic)


    
