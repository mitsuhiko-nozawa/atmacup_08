from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold
import pandas as pd
import numpy as np
class KFManager():
    def __init__(self, param, module):
        self.param = param
        self.X = module["X"]
        self.y = module["y"]
        self.g = self.X["Publisher"]
        self.seed = self.param["seed"]
        self.validation = self.param["validation"]
        self.train_indices = []
        self.valid_indices = []
        self.make_KFold(self.validation)

    def make_KFold(self, validation):
        if validation == "GroupKFold":
            n_splits = self.param["K"]
            #kf = GroupShuffleSplit(n_splits=n_splits, random_state=self.seed, test_size=1/n_splits, train_size=1-1/n_splits)
            kf = PublidherGroupKFold(n_splits=n_splits, random_state=self.seed, shuffle=True)
            for tr_ind, val_ind in kf.split(self.X, self.y, self.g):
                self.train_indices.append(tr_ind)
                self.valid_indices.append(val_ind)

        elif validation == "KFold":
            kf = KFold(n_splits=self.param["K"], random_state=self.seed, shuffle=True)
            for tr_ind, val_ind in kf.split(self.X):
                self.train_indices.append(tr_ind)
                self.valid_indices.append(val_ind)

        elif validation == "StratifiedKFold":
            kf = binningStratifiedKFold(n_splits=self.param["K"], random_state=self.seed, shuffle=True)
            for tr_ind, val_ind in kf.split(self.X, self.y):
                self.train_indices.append(tr_ind)
                self.valid_indices.append(val_ind)
            

        del self.X, self.y, self.g

    def split(self):
        for train_index, valid_index in zip(self.train_indices, self.valid_indices):
            yield train_index, valid_index
    

class PublidherGroupKFold():
    def __init__(self, n_splits, random_state, shuffle):
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    def split(self, X, y, g):
        g = g.unique()
        for tr_g_ind, val_g_ind in self.kf.split(g):
            tr_g = g[tr_g_ind]
            val_g = g[val_g_ind]
            train_index = X[X["Publisher"].isin(tr_g)].index.to_list()
            valid_index = X[X["Publisher"].isin(val_g)].index.to_list()
            yield train_index, valid_index

class binningStratifiedKFold():
    def __init__(self, n_splits, random_state, shuffle):
        self.kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y):
        num_bins = np.int(1 + np.log2(len(X)))
        bins = pd.cut(
            y,
            bins=num_bins,
            labels=False
        )
        for train_index, valid_index in self.kf.split(X, bins):
            yield train_index, valid_index

