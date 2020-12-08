import numpy as np
import pandas as pd
import sys, os
import os.path as osp
import random

from sklearn.metrics import mean_squared_log_error

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def make_data(param):
    dtypes = {
        #"Name" : "object",
        #"Platform" : "object",
        #"User_Score" : "float32",
        #"Critic_Score" : "float32",
    }

    train_df = pd.read_csv(osp.join(param["data_path"], param["train_file"]), dtype=dtypes)
    test_df = pd.read_csv(osp.join(param["data_path"], param["test_file"]), dtype=dtypes)

    module = dict(
        train_df = train_df,
        test_df = test_df,
    )
    return module


def pick_data(module):
    train_df = module["train_df"]
    test_df = module["test_df"]
    all_df = train_df.append(test_df)
    return train_df, test_df, all_df


def save_module(module, all_df, train_size):
    train_df = all_df[:train_size]
    test_df = all_df[train_size:]
    module["train_df"] = train_df
    module["test_df"] = test_df

    return module


def metric(y_true, y_preds):
    return mean_squared_log_error(y_true, y_preds)**0.5