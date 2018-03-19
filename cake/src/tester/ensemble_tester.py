# coding:utf-8

import glob

from cake.src.tester.basic_tester import BasicTester

def filter_model_paths(model_folder):
    file_paths = []
    clf_path = ""
    for file_path in glob.glob(model_folder):
        if file_path.endswith(".pkl"):
            clf_path = file_path
        elif "checkpoint" not in file_path:
            file_paths.append("%s.ckpt" % file_path.split(".")[0])
    model_paths = list(set(file_paths))
    return clf_path, model_paths
    

class Ensemble_Tester(object):
    
    def __init__(self, model_folder, result_type="label", method="vote"):
        self.model_folder = model_folder
        self.result_type = result_type
        self.method = method
        self.clf_path, self.model_paths = filter_model_paths(self.model_folder)
        self.ensemble_models = [BasicTester(model_path, None, self.result_type) for model_path in self.model_paths]
        
    def test(self, input_data):
        pass
