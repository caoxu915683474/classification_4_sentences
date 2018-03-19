# coding:utf-8

import os
import sys
import numpy as np
import tensorflow as tf
import keras as K 
import pandas as pd
import sklearn.linear_model as lm
from collections import Counter
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from cake.src.trainer.ensemble_trainer import Ensemble_Trainer
from cake.src.tester.basic_tester import BasicTester

def average_stacking(input_data, real_label, output_folder):
    pred_y = []
    n_of_clfs = len(input_data)
    compound_probs = np.sum(input_data, axis=0)/n_of_clfs
    for i, d in enumerate(compound_probs):
        pred_y.append(np.argmax(d))
    matrix = confusion_matrix(real_label, pred_y)
    # TODO: Change to multi-classification 
    print("Confusion matrix of stacked model: \n")
    print(matrix)

def vote_stacking(input_data, real_label, output_folder):
    pred_y = []
    for i, d in enumerate(input_data):
        c = Counter(d)
        pred_y.append(c.most_common(1)[0][0])
    matrix = confusion_matrix(real_label, pred_y)
    # TODO: Change to multi-classification 
    print("Confusion matrix of stacked model: \n")
    print(matrix)

def logistic_stacking(input_data, real_label, output_folder):
    clf = lm.LogisticRegressionCV(multi_class="ovr", fit_intercept=True, cv=2,penalty="l2", solver="lbfgs", tol=0.01) 
    X_train, X_test, Y_train, Y_test = train_test_split(input_data, real_label, test_size=0.3, random_state=0)
    clf.fit(X_train,Y_train)
    pred_y = clf.predict(X_test)
    matrix = confusion_matrix(Y_test, pred_y)
    # TODO: Change to multi-classification 
    print("Confusion matrix of stacked model: \n")
    print(matrix)
    print("Savinf model to %s./clf.pkl" % output_folder)
    joblib.dump(clf, "%s./clf.pkl" % output_folder)
    print("")

def linear_stacking(input_data, real_label, output_folder):
    clf = lm.LinearRegression()
    X_train, X_test, Y_train, Y_test = train_test_split(input_data, real_label, test_size=0.3, random_state=0)
    clf.fit(X_train)
    pred_y = clf.predict(X_test)
    matrix = confusion_matrix(Y_test, pred_y)
    print("Confusion matrix of stacked model: \n")
    print(matrix)
    print("Model save to %s./clf.pkl" % output_folder)
    joblib.dump(clf, "%s./clf.pkl" % output_folder)
    print("")

class Stack_Trainer(Ensemble_Trainer):

    def data_helper(self, result_type="label"):
        """
        Restore the models one by one and create the input data for stacking model.
        @ param `result_type` denote which kind of result will be used, "label" result or "probs".
        @ return input data
        """
        predict_result = []
        for m in self._model_paths:
            tf.reset_default_graph()
            print("Loading model from %s" % m)
            tester = BasicTester(m, self.output_folder, result_type)
            predict_result.append(tester.test(self.raw_data))
            # TODO:  Using evaluate function, Print the accuracy of each model
        self.input_data = np.array(predict_result).T.tolist() if result_type=="label" else predict_result

    def train(self, result_type="label", stack_method="logistic"):
        """
        Train the stacked model.
        @ param `stack_method` method to stack the models, "logistic" "linear" "average" "vote"
        """

        if result_type == "label":
            if stack_method not in ["logistic", "linear", "vote"]:
                print("Result_type and stack_method not match.")
                exit(1)
        elif result_type == "probs":
            if stack_method != "average":
                print("Result_type and stack_method not match.")
                exit(1)

        print("Preparing the input data for stacking model.")
        self.data_helper(result_type=result_type)
        print("Data preparation done.")

        dic = {"input_data":self.input_data, "real_label":self.real_label, "output_folder": self.output_folder}

        print("Start training the stacking model using %s" % stack_method)
        return getattr(sys.modules[__name__], "%s_stacking" % stack_method)(**dic)

