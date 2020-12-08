import pandas as pd
import os.path as osp


class Logger():
    """
    result.csvに書き込み
    """

    def __init__(self, param):
        self.param = param

    def __call__(self, module):
        ############################################################
        if self.param["make_submit"]["flag"]:
            sub_df = module["test_X"]
            sub_df[self.param["target_col"]] = module["test_y"]
            for col in sub_df.columns:
                if col != self.param["target_col"]:
                    sub_df.drop(col, axis=1, inplace=True)

            save_path = osp.join(self.param["dir_path"], "submission.csv")
            sub_df.to_csv(save_path, index=False)
        ############################################################
        if self.param["save_X"]["flag"]:
            X = module["X"]
            save_path = osp.join(self.param["dir_path"], "X.csv")
            X.to_csv(save_path, index=False)

        ############################################################
        if self.param["make_train_preds"]["flag"]:
            train_preds = pd.DataFrame(module["train_preds"], columns=["Global_Sales"])
            save_path = osp.join(self.param["dir_path"], "train_preds.csv")
            train_preds.to_csv(save_path, index=False)

        ############################################################
        result_path = osp.join(self.param["dir_path"], "result.csv")
        try:
            result_df = pd.read_csv(result_path)
        except:
            result_df = pd.DataFrame()
        param = dict()
        param.update(self.param["model_param"])
        param["seed"] = len(self.param["seeds"])
        param["fold_num"] = self.param["K"]
        param["cv"] = module["err"]
        param["features"] = ', '.join(module["X"].columns.to_list())


        result_df = result_df.append(
            pd.DataFrame(param, index=[0,])
        )

        result_df.to_csv(result_path, index=False)









