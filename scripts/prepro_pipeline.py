import sys
import os
import csv
import yaml
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
sys.path.append("./")

import cake.src.config as config
from cake.src.preprocess.loader_toxic_comment import ToxicCommentLoader
from cake.src.preprocess.preprocesser_en import EnPreprocesser
from cake.src.preprocess.preprocesser_zh import ZhPreprocesser
from cake.src.analyse.feature_engineering.feature_engineer import FeatureEngineer
from cake.src.analyse.basic_analyse import BasicAnalyse
from cake.src.utils.batch_manager import BatchManager
from cake.src.utils.data_helper import *
from cake.src.trainer.basic_trainer import BasicTrainer
from cake.src.tester.basic_tester import BasicTester

# Get the opts from command line
parser = argparse.ArgumentParser(description='Some usage...')
parser.add_argument("--config_file", type=str, help="location of the configuration file.")
parser.add_argument("--input_file", type=str, help="location of the configuration file.")
parser.add_argument("--output_folder", type=str, help="location to save the analyse result.")
Flags = parser.parse_args()

def verify_configurations(config):
    pipeline = config[0]
    assert pipeline["module"] == "pipeline", "Please indicate which module will be done."
    assert list(pipeline["pipeline"].values()).count(0) < 6, "At least one module should be chosen."
    modules = [c["module"] for c in config]
    for p, v in zip(pipeline.keys(), pipeline.values()):
        if v == 1:
            assert p in modules, "Please provide %s params" % p
            if p in ["test", "ensemble_test"]:
                assert "preprocess" in modules, "Please provide preprocess params for %s" % p
    return True

def load_configs(config_file):
    assert os.path.isfile(Flags.config_file), "Please provide correct configuration file path"
    config = yaml.load(open(config_file, "r"))
    assert verify_configurations(config), "Configuration error."
    return config

def find_config(config, module_name):
    for c in config:
        if c["module"] == module_name:
            return c

def print_configs(c, module_name):
    print("\nConfiguration: %s" % module_name)
    print(yaml.dump(c, indent=4, default_flow_style=False, allow_unicode=True))
    print("")

def preprocess(config):
    config = find_config(config, "preprocess")
    print_configs(config, "Preprocess")
    assert os.path.isfile(config["input_file"]), "Please provide correct corpus file path."
    train = pd.read_csv(config["input_file"], converters={"comment":str})

    # Preprocess
    if config["params"]["lang"] == "en":
        preprocesser = EnPreprocesser(**config["params"]["en"])
    else:
        preprocesser = ZhPreprocesser(**config["params"]["zh"])
    train["comment"] = preprocesser.preprocess_corpus(train["comment"].tolist())
    train.to_csv(config["output_file"], quoting=csv.QUOTE_NONNUMERIC, index=False)

def analyse(config):
    config = find_config(config, "analyse")
    print_configs(config, "Analyse")
    assert os.path.isfile(config["input_file"]), "Please provide correct preprocessed corpus file path."
    assert os.path.isdir(config["output_dir"]), "Please provide correct output folder path."

    # TODO: Valid check should be put in verify_config()
    label_type = int(config["basic"]["label_type"]) if "label_type" in config else 0
    balance_or_not = int(config["basic"]["balance_or_not"]) if "balance_or_not" in config else 0

    analyser = BasicAnalyse()
    analyser.analyse(
        config["input_file"],
        config["output_dir"],
        balance_or_not = balance_or_not,
        label_type=label_type
    )
   
    feature_opt = FeatureEngineer(**config["params"]["feature_engineer"])
    feature_opt.run(
        corpus_path = config["input_file"],
        report_path = config["output_dir"],
        k=20
    )
    
def train(config):
    config = find_config(config, "train")
    print_configs(config, "Train")
   
    assert os.path.isfile(config["params"]["data_helper"]["input_file"]), "Input file not exists!"
    assert os.path.isfile(config["params"]["data_helper"]["word_emb_path"]), "Word embedding or word_2_id file not exists!"

    print("Generating batches from file...")
    bm = BatchManager(**config["params"]["data_helper"])
    train_batches = bm.iter_on_batch("train", num_epochs=50)
    dev_X, dev_Y = bm.iter_on_file("dev")

    print("Loading model with config...")
    trainer = BasicTrainer(**{**config["params"]["model"], **bm.configs})
    
    print("Start training...")
    for step, batch in enumerate(train_batches):
        x_batch, y_batch = batch
        trainer.train_step(x_batch, y_batch, step)

        if (step % config["params"]["data_helper"]["evaluate_every"] == 0) and \
          (trainer.dev_step(dev_X, dev_Y, step, config["params"]["data_helper"]["early_stop"])):
            break

    print("Cleaning up trainer...")
    trainer.close_session()

def test(config):
    prepro_config = find_config(config, "preprocess")
    test_config = find_config(config, "test")
    print_configs(test_config, "Test")
  
    assert os.path.isfile(test_config["params"]["input_file"]), "Input file not exists!"
    assert os.path.isfile(test_config["params"]["data_helper"]["word_emb_path"]), "Word embedding or word_2_id file not exists!"

    print("Preprocessing data..")
    test = pd.read_csv(test_config["params"]["input_file"], converters={"comment":str})
    if prepro_config["params"]["lang"] == "en":
        preprocesser = EnPreprocesser(**prepro_config["params"]["en"])
    else:
        preprocesser = ZhPreprocesser(**prepro_config["params"]["zh"])
    test["comment"] = preprocesser.preprocess_corpus(test["comment"].tolist())

    print("Generating data for test...")
    raw_data = data_helper(
        list(test["comment"]),
        **test_config["params"]["data_helper"])

    print("Inferencing...")
    tester = BasicTester(model_path = test_config["params"]["model"]["model_path"])
    pred_y = tester.test(raw_data)
    
    print("Saving the inference result to %s" % test_config["params"]["output_file"])
    test["pred_label"] = pred_y
    test.to_csv(test_config["params"]["output_file"], quoting=csv.QUOTE_NONNUMERIC, index=False)
    
def ensemble_train(config):
    config = find_config(config, "ensemble_train")
    print_configs(config, "Ensemble_Train")

    assert "ensemble_method" in config["params"], "Please provide ensemble method!"
    assert os.path.isfile(config["params"]["input_file"]), "Input file not exists!"
    assert os.path.isfile(config["params"]["data_helper"]["word_emb_path"]), "Word embedding or word_2_id file not exist!"

    print("Loading train data...")
    train = pd.read_csv(config["params"]["input_file"], converters={"comment":str})

    print("Generating data for %s train" % config["params"]["ensemble_method"])
    raw_data = data_helper(
        list(train["comment"]),
        **config["params"]["data_helper"])
    labels = list(train["label"])

    print("Start ensemble training...")
    stack_method = config["params"]["ensemble_method"]
    _module = __import__("cake.src.trainer.%s_trainer" % stack_method.lower(), fromlist=["%s_Trainer" % stack_method])
    param_dic = {
        "model_paths":config["params"]["model"]["model_paths"],
        "raw_data":raw_data,
        "real_label":labels,
        "output_folder":config["params"]["model"]["output_folder"]}
    stack_trainer = getattr(_module, "%s_Trainer" % stack_method)(**param_dic)
    stack_trainer.train(config["params"]["model"]["result_type"], config["params"]["model"]["method"])
    
def ensemble_test(config):
    prepro_config = find_config(config, "preprocess")
    test_config = find_config(config, "ensemble_test")
    print_configs(test_config, "Ensemble_Test")

    assert "ensemble_method" in test_config["params"], "Please provide ensemble method!"
    assert os.path.isfile(test_config["params"]["input_file"]), "Please provide correct test data"
    assert os.path.isfile(test_config["params"]["data_helper"]["word_emb_path"]), "Word embedding or word_2_id file not exist!"

    print("Process data for ensemble test")
    data = pd.read_csv(test_config["params"]["input_file"], converters={"comment":str})
    if prepro_config["params"]["lang"] == "en":
        preprocesser = EnPreprocesser(**prepro_config["params"]["en"])
    else:
        preprocesser = ZhPreprocesser(**prepro_config["params"]["zh"])

    # Data Helper
    print("Generating data for test")
    raw_data = data_helper(
        preprocesser.preprocess_corpus(data["comment"].tolist()),
        **test_config["params"]["data_helper"])
    
    # Test
    print("Start ensemble testing...")
    stack_method = test_config["params"]["ensemble_method"]
    _module = __import__("cake.src.tester.%s_tester" % stack_method.lower(), fromlist=["%s_Tester" % stack_method])
    ensemble_tester = getattr(_module, "%s_Tester" % stack_method)(**test_config["params"]["model"])

    print("Saving the inference result to %s" % test_config["params"]["output_file"])
    pred_y = ensemble_tester.test(raw_data)
    data["pred_y"] = pred_y
    data.to_csv(test_config["params"]["output_file"], quoting=csv.QUOTE_NONNUMERIC, index=False)

def main():
    config = load_configs(Flags.config_file)

    pipe_line = [m for m,v in zip(config[0]["pipeline"].keys(), config[0]["pipeline"].values()) if v == 1]
    print(" ===> ".join(pipe_line))

    for module_name in pipe_line:
        getattr(sys.modules[__name__], module_name)(config)

if __name__ == "__main__":
    
    # # Prepare Data
    # print("Loading data...")
    # ToxicCommentLoader(
    #     train_path="/home/cxpc/Documents/nlp/Text_Classification/data/data_toxic-comment/train.csv",
    #     test_path="/home/cxpc/Documents/nlp/Text_Classification/data/data_toxic-comment/test.csv",
    #     output_path="./data/"
    # ).load()

    main()
