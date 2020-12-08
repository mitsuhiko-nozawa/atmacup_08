import os.path as osp
import numpy as np
import pickle

from utils import seed_everything, metric
from validation import KFManager

import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error



class Train():
    def __init__(self, param):
        self.param = param

    def __call__(self, module):
        train_preds = []
        errs = []

        for seed in self.param["seeds"]:
            oof_preds, err = self.train_by_seed(seed, module)
            train_preds.append(oof_preds)
            errs.append(err)
        train_preds = np.mean(np.array(train_preds), axis=0)
        err = metric(module["y"], train_preds)
        print(f"final RMSLE is {err}")
        module["train_preds"] = train_preds
        module["errs"] = errs
        module["err"] = err
        return module


    def train_by_seed(self, seed, module):
        seed_everything(seed)
        self.param["seed"] = seed
        self.param["model_param"]["random_state"] = seed
        kf_manager = KFManager(self.param, module)

        X = module["X"]
        y = module["y"]
        print(X.columns)
        oof_preds = np.zeros((X.shape[0], ))

        if self.param["metric"] == "RMSLE":
            y = np.log1p(y)
        for fold, (train_index, valid_index) in enumerate(kf_manager.split()):
            train_X = X.loc[train_index].drop("Publisher", axis=1).values
            valid_X = X.loc[valid_index].drop("Publisher", axis=1).values
            train_y = y.loc[train_index].values
            valid_y = y.loc[valid_index].values
            print(train_X.shape)
            print(valid_X.shape)
            oof = self.train_by_fold(train_X, train_y, valid_X, valid_y, self.param, fold)
            oof_preds[valid_index] += oof

        if self.param["metric"] == "RMSLE":
            oof_preds = np.expm1(oof_preds)
            y = np.expm1(y)
        
        err = metric(y, oof_preds)
        print(f"seed {seed} : RMSLE is {err}")
        return oof_preds, err

    def train_by_fold(self, train_X, train_y, valid_X, valid_y, param, fold):
        if param["model"] == "LGBM":
            oof = LGBM_train(train_X, train_y, valid_X, valid_y, param, fold)
            #print(metric(np.expm1(valid_y), np.expm1(oof)))
        return oof
        

def LGBM_train(train_X, train_y, valid_X, valid_y, param, fold):
    train_data = lgb.Dataset(train_X, label=train_y)
    valid_data = lgb.Dataset(valid_X, label=valid_y)
    model = lgb.train(
        param["model_param"], 
        train_data, 
        valid_sets=valid_data, 
        early_stopping_rounds=param["early_stopping_rounds"], 
        verbose_eval=param["verbose_eval"],

    )
    pred = model.predict(valid_X)
    pred = np.where(pred < 0, 0, pred)
    fname = f"{param['seed']}_{fold}.pkl"

    path = osp.join(param["dir_path"], "weight/", fname)
    pickle.dump(model, open(path, 'wb'))
    return pred
    