import os
import os.path as osp
import numpy as np
import pickle

class Infer():
    def __init__(self, param):
        self.param = param

    def __call__(self, module):
        weight_files = os.listdir(osp.join(self.param["dir_path"], "weight/"))
        preds = []
        test_X = module["test_X"]
        if "Publisher" in test_X.columns:
            test_X.drop("Publisher", axis=1, inplace=True)

        #test_X = test_X[['Critic_Score','Critic_Count', 'User_Score', 'User_Count','Year_of_Release']]

        test_y = []
        for file in weight_files:
            print(file)
            path = osp.join(self.param["dir_path"], "weight/",file)
            if self.param["model"] == "LGBM":
                preds = infer_lightGBM(test_X, path)
                if self.param["metric"] == "RMSLE":
                    preds = np.expm1(preds)
                test_y.append(preds)
        test_y = np.mean(np.array(test_y), axis=0)
        module["test_y"] = test_y
        return module



def infer_lightGBM(test_X, path):
    model = pickle.load(open(path, 'rb'))
    res = model.predict(test_X)
    return np.where(res < 0, 0, res)


