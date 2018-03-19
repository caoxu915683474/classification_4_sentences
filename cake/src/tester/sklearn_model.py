# coding:utf-8

import sklearn.linear_model as lm
from sklearn.externals import joblib

class Sklearn_Model(object):
    """
    Implement the sklearn model of stacking.
    """
    def __init__(self, model_path, output_folder=None, result_type="label"):
        """
        Constructor.
        @ param `model_path` location of the model.
        @ param `output_folder` where to save the stacked model.
        @ param `result_type` "label" or "probs"
        """
        self.model_path = model_path
        self.output_folder = output_folder
        self.result_type = result_type

    def test(self, raw_data):
        model = joblib.load(self.model_path)
        result = model.predict(raw_data) if self.result_type=="label" else model.predict_proba(raw_data)
        if self.output_folder != None:
            joblib.dump(model, "%s./clf.pkl" % self.output_folder)
        return result
