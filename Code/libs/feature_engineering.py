import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import pick_data, save_module

def make_func(func_name):
    if func_name == "User_Count_tbd2Null":
        return User_Count_tbd2Null
    elif func_name == "fillna":
        return fillna
    elif func_name == "label_encode":
        return label_encode
    elif func_name == "make_X_y":
        return make_X_y
    elif func_name == "Summarize_Sales":
        return Summarize_Sales
    elif func_name == "count_encode":
        return count_encode
    elif func_name == "onehot_encode":
        return onehot_encode
    elif func_name == "equal_Pub_Dev":
        return equal_Pub_Dev
    elif func_name == "Publisher_encode":
        return Publisher_encode
    elif func_name == "Developer_encode":
        return Developer_encode
    elif func_name == "Year_of_Release_encode":
        return Year_of_Release_encode
    elif func_name == "Rating_encode":
        return Rating_encode
    elif func_name == "is_Release_Year_of_Platform":
        return is_Release_Year_of_Platform
    elif func_name == "Prod":
        return Prod

def Prod(module, param):
    if param["flag"]:
        train_df, test_df, all_df = pick_data(module)
        for cols in param["cols"]:
            name = " * ".join(cols)
            all_df[name] = ""
            for i, col in enumerate(cols):
                if i != 0:
                    all_df[name] = all_df[name] + " * "
                all_df[name] = all_df[name] +  all_df[col].astype("str").fillna("Null")
        return save_module(module, all_df, train_df.shape[0])
    else :
        return module


def is_Release_Year_of_Platform(module, param):
    if param["flag"]:
        train_df, test_df, all_df = pick_data(module)
        gb = all_df.groupby("Platform")
        min_Y = gb["Year_of_Release"].min()
        all_df["is_Release_Year_of_Platform"] = all_df["Platform"].map(min_Y)
        all_df["is_Release_Year_of_Platform"] = (all_df["is_Release_Year_of_Platform"] == all_df["Year_of_Release"]).astype(int)
        return save_module(module, all_df, train_df.shape[0])
    else :
        return module

def equal_Pub_Dev(module, param):
    if param["flag"]:
        train_df, test_df, all_df = pick_data(module)
        all_df["equal_Pub_Dev"] = all_df.apply(lambda x : 1 if x["Publisher"] == x["Developer"] else 0, axis=1)
        return save_module(module, all_df, train_df.shape[0])
    else :
        return module

def fillna(module, param):
    if param["flag"]:
        train_df, test_df, all_df = pick_data(module)
        null_cols = all_df.T[all_df.isnull().any()].index.to_list()
        for col in null_cols:
            if all_df[col].dtype in [int, float]:
                all_df[col].fillna(-999, inplace=True)
            else:
                all_df[col].fillna("missing value", inplace=True)
        return save_module(module, all_df, train_df.shape[0])
    else :
        return module


def User_Count_tbd2Null(module, param):
    if param["flag"]:
        train_df, test_df, all_df = pick_data(module)
        #all_df["is_tbd"] = all_df["User_Score"].map(lambda x : 1 if x == "tbd" else 0)
        all_df["User_Score"] = all_df['User_Score'].replace('tbd', None).astype(float)
        return save_module(module, all_df, train_df.shape[0])
    else :
        return module


def label_encode(module, param):
    
    #attribute
    #module : dict of data
    #cols : columns to encode 

    #return 
    #module (updated)
    
    if param["flag"]:
        train_df, test_df, all_df = pick_data(module)
        cols = param["cols"] + [col for col in all_df.columns if all_df[col].dtype not in [int, float]]
        for col in cols:
            all_df[col] = LabelEncoder().fit_transform(all_df[col].values.reshape(-1,))
        return save_module(module, all_df, train_df.shape[0])
    else :
        return module

def onehot_encode(module, param):
    if param["flag"]:
        train_df, test_df, all_df = pick_data(module)
        for col in param["cols"]:
            vc = all_df[col].dropna().value_counts()
            cats = vc[vc > 80].index

            x = pd.Categorical(all_df[col], categories=cats)
            out_df = pd.get_dummies(x, dummy_na=False)
            out_df.columns = out_df.columns.tolist()   
            out_df.add_prefix(f'OH_{col}=')  
            all_df = pd.concat([all_df, out_df], axis=1)       
        return save_module(module, all_df, train_df.shape[0])
    else :
        return module

def make_X_y(module, param):
    if param["flag"]:
        train_df, test_df = module["train_df"], module["test_df"]
        #test_df = module["test_df"]
        module["y"] = train_df[param["target_col"]]
        for col in param["drop_cols"]:
            for df in [train_df, test_df]:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)
        module["X"] = train_df
        module["test_X"] = test_df 
        del module["train_df"], module["test_df"]

    return module


def Summarize_Sales(module, param):
    if param["flag"]:
        train_df, test_df, all_df = pick_data(module)
        sales_cols = param["sales_cols"]
        bys = param["by"]
        for by in bys:
            # mean
            summary_mean_df = train_df.groupby(by)[sales_cols].mean()
            summary_mean_df = summary_mean_df.apply(lambda x : x / x["Global_Sales"], axis=1).drop("Global_Sales", axis=1)
            all_df = all_df.merge(summary_mean_df, on=by, suffixes=("", f"_Mean_by_{by}"), how="left")

            # max
            summary_max_df = train_df.groupby(by)[sales_cols].max()
            all_df = all_df.merge(summary_max_df, on=by, suffixes=("", f"_Max_by_{by}"), how="left")

            # var
            summary_var_df = train_df.groupby(by)[sales_cols].var()
            all_df = all_df.merge(summary_var_df, on=by, suffixes=("", f"_Var_by_{by}"), how="left")    

            # sum
            #summary_sum_df = train_df.groupby(by)[sales_cols].sum()
            #all_df = all_df.merge(summary_sum_df, on=by, suffixes=("", f"_Sum_by_{by}"), how="left") 

            # skew
            #summary_skew_df = train_df.groupby(by)[sales_cols].skew()
            #sall_df = all_df.merge(summary_skew_df, on=by, suffixes=("", f"_Skew_by_{by}"), how="left")  



        return save_module(module, all_df, train_df.shape[0])
    else :
        return module


def count_encode(module, param):
    if param["flag"]:
        train_df, test_df, all_df = pick_data(module)
        for col in param["cols"]:
            col_name = f"ce_{col}"
            vc = all_df[col].value_counts()
            all_df[col_name] = all_df[col].map(vc)
            all_df.drop(col_name, axis=1)
        return save_module(module, all_df, train_df.shape[0])
    else :
        return module

def Publisher_encode(module, param):
    if param["flag"]:
        train_df, test_df, all_df = pick_data(module)
        # NameにPSが含まれるかどうか
        all_df["is_PS"] = all_df["Platform"].map(lambda x : int("PS" in str(x)))
        all_df["is_PC"] = all_df["Platform"].map(lambda x : int("PC" == str(x)))

        vc = all_df["Publisher"].value_counts()
        g = all_df.groupby("Publisher")

        # 出しているゲームのうち、NameにPSが含まれる割合
        col = "is_PS"
        all_df["Pub_PS_rate"] = all_df["Publisher"].map(g[col].sum() / vc)

        # 出しているゲームのうち、PlatformにPCが含まれる割合
        col = "is_PC"
        all_df["Pub_PC_rate"] = all_df["Publisher"].map(g[col].sum() / vc)
        #all_df.drop(col, axis=1, inplace=True)

        # データセットに存在する年代の広さ
        col = "Year_of_Release"
        #all_df["Pub_Spread_of_Generation"] = all_df["Publisher"].map(g[col].max() - g[col].min())

        # データセットに存在する年代の広さ
        #col = "Year_of_Release"
        all_df["Pub_Number_of_Generation"] = all_df["Publisher"].map(g[col].count())

        # 出している作品のうち、レビューがついているものの割合, 統計情報(User)
        col = "User_Count"
        #all_df["Pub_Reviewed_Rate"] = all_df["Publisher"].map(g[col].count() / vc)#
        #all_df["Pub_User_Count_Sum"] = all_df["Publisher"].map(g[col].sum())
        #all_df["Pub_User_Count_Mean"] = all_df["Publisher"].map(g[col].mean())
        #all_df["Pub_User_Count_Median"] = all_df["Publisher"].map(g[col].median())#
        #all_df["Pub_User_Count_Max"] = all_df["Publisher"].map(g[col].max())
        #all_df["Pub_User_Count_Min"] = all_df["Publisher"].map(g[col].min())#
        #all_df["Pub_User_Count_Std"] = all_df["Publisher"].map(g[col].std())#
        #all_df["Pub_User_Count_Skew"] = all_df["Publisher"].map(g[col].skew())

        ########################################################################################
        # 出している作品のうち、レビュースコアの統計情報(User)
        col = "User_Score"
        temp_g = all_df[all_df["User_Score"] != "tbd"]; temp_g["User_Score"] = temp_g["User_Score"].astype(float)
        temp_g = temp_g.groupby("Publisher")
        all_df["Pub_User_Score_Sum"] = all_df["Publisher"].map(temp_g[col].sum())#
        #all_df["Pub_User_Score_Mean"] = all_df["Publisher"].map(temp_g[col].mean())
        #all_df["Pub_User_Score_Max"] = all_df["Publisher"].map(temp_g[col].max())
        #all_df["Pub_User_Score_Median"] = all_df["Publisher"].map(temp_g[col].median())
        #all_df["Pub_User_Score_Min"] = all_df["Publisher"].map(temp_g[col].min())
        #all_df["Pub_User_Score_Skew"] = all_df["Publisher"].map(temp_g[col].skew())

        ########################################################################################
        # 出している作品のうち、レビューがついているものの割合, 統計情報(Critic)
        col = "Critic_Count"
        #all_df["Pub_Critic_Count_Sum"] = all_df["Publisher"].map(g[col].sum())#
        #all_df["Pub_Critic_Count_Mean"] = all_df["Publisher"].map(g[col].mean())
        #all_df["Pub_Critic_Count_Median"] = all_df["Publisher"].map(g[col].median())
        #all_df["Pub_Critic_Count_Max"] = all_df["Publisher"].map(g[col].max())#
        #all_df["Pub_Critic_Count_Min"] = all_df["Publisher"].map(g[col].min())#
        #all_df["Pub_Critic_Count_Std"] = all_df["Publisher"].map(g[col].std())
        #all_df["Pub_Critic_Count_Skew"] = all_df["Publisher"].map(g[col].skew())        

        ########################################################################################
        # 出している作品のうち、レビュースコアの統計情報(Critic)
        col = "Critic_Score"
        all_df["Pub_Critic_Score_Sum"] = all_df["Publisher"].map(g[col].sum())#
        all_df["Pub_Critic_Score_Mean"] = all_df["Publisher"].map(g[col].mean())#
        all_df["Pub_Critic_Score_Median"] = all_df["Publisher"].map(g[col].median())#
        all_df["Pub_Critic_Score_Max"] = all_df["Publisher"].map(g[col].max())#
        #all_df["Pub_Critic_Score_Min"] = all_df["Publisher"].map(g[col].min())
        #all_df["Pub_Critic_Score_Std"] = all_df["Publisher"].map(g[col].std())
        all_df["Pub_Critic_Score_Skew"] = all_df["Publisher"].map(g[col].skew())#

        ########################################################################################

        #出しているゲームの数
        all_df["Pub_Game_Num"] = all_df["Publisher"].map(vc)

        # 展開しているPlatformの数
        col = "Platform"
        all_df["Pub_Plat_Num"] = all_df["Publisher"].map(g[col].unique().map(lambda x : len(x)))

        return save_module(module, all_df, train_df.shape[0])
    else :
        return module


def Developer_encode(module, param):
    if param["flag"]:
        train_df, test_df, all_df = pick_data(module)
        # NameにPSが含まれるかどうか
        all_df["is_PS"] = all_df["Platform"].map(lambda x : int("PS" in str(x)))
        all_df["is_PC"] = all_df["Platform"].map(lambda x : int("PC" == str(x)))

        vc = all_df["Developer"].value_counts()
        g = all_df.groupby("Developer")

        # 出しているゲームのうち、NameにPSが含まれる割合
        col = "is_PS"
        all_df["Dev_PS_rate"] = all_df["Developer"].map(g[col].sum() / vc)

        # 出しているゲームのうち、PlatformにPCが含まれる割合
        col = "is_PC"
        all_df["Dev_PC_rate"] = all_df["Developer"].map(g[col].sum() / vc)
        #all_df.drop(col, axis=1, inplace=True)
        # データセットに存在する年代の広さ
        col = "Year_of_Release"
        all_df["Dev_Spread_of_Generation"] = all_df["Developer"].map(g[col].max() - g[col].min())

        ########################################################################################
        # 出している作品のうち、レビューカウントの統計情報(User)
        col = "User_Count"
        #all_df["Dev_Reviewed_Rate"] = all_df["Developer"].map(g[col].count() / vc)#
        #all_df["Dev_User_Count_Sum"] = all_df["Developer"].map(g[col].sum())
        #all_df["Dev_User_Count_Mean"] = all_df["Developer"].map(g[col].mean())#
        #all_df["Dev_User_Count_Median"] = all_df["Developer"].map(g[col].median())#
        #all_df["Dev_User_Count_Max"] = all_df["Developer"].map(g[col].max())
        #all_df["Dev_User_Count_Min"] = all_df["Developer"].map(g[col].min())
        #all_df["Dev_User_Count_Std"] = all_df["Developer"].map(g[col].std())
        #all_df["Dev_User_Count_Skew"] = all_df["Developer"].map(g[col].skew())


        ########################################################################################
        # 出している作品のうち、レビューの平均(User)
        col = "User_Score"
        temp_g = all_df[all_df["User_Score"] != "tbd"]; temp_g["User_Score"] = temp_g["User_Score"].astype(float)
        temp_g = temp_g.groupby("Developer")
        #all_df["Dev_User_Score_Sum"] = all_df["Developer"].map(temp_g[col].sum())
        all_df["Dev_User_Score_Mean"] = all_df["Developer"].map(temp_g[col].mean())#
        #all_df["Dev_User_Score_Max"] = all_df["Developer"].map(temp_g[col].max())
        #all_df["Dev_User_Score_Median"] = all_df["Developer"].map(temp_g[col].median())
        #all_df["Dev_User_Score_Min"] = all_df["Developer"].map(temp_g[col].min())
        #all_df["Dev_User_Score_Skew"] = all_df["Developer"].map(temp_g[col].skew())

        ########################################################################################
        # 出している作品のうち、レビューがついているものの割合, 統計情報(Critic)
        col = "Critic_Count"
        #all_df["Dev_Critic_Count_Sum"] = all_df["Developer"].map(g[col].sum())
        #all_df["Dev_Critic_Count_Mean"] = all_df["Developer"].map(g[col].mean())
        #all_df["Dev_Critic_Count_Median"] = all_df["Developer"].map(g[col].median())#
        #all_df["Dev_Critic_Count_Max"] = all_df["Developer"].map(g[col].max())
        #all_df["Dev_Critic_Count_Min"] = all_df["Developer"].map(g[col].min())#
        #all_df["Dev_Critic_Count_Std"] = all_df["Developer"].map(g[col].std())#
        #all_df["Dev_Critic_Count_Skew"] = all_df["Developer"].map(g[col].skew())#

        ########################################################################################
        # 出している作品のうち、レビューの統計情報(Critic))
        col = "Critic_Score"
        all_df["Dev_Critic_Score_Sum"] = all_df["Developer"].map(g[col].sum())#
        all_df["Dev_Critic_Score_Mean"] = all_df["Developer"].map(g[col].mean())#
        all_df["Dev_Critic_Score_Median"] = all_df["Developer"].map(g[col].median())#
        all_df["Dev_Critic_Score_Max"] = all_df["Developer"].map(g[col].max())#
        all_df["Dev_Critic_Score_Min"] = all_df["Developer"].map(g[col].min())#
        #all_df["Dev_Critic_Score_Std"] = all_df["Developer"].map(g[col].std())
        #all_df["Dev_Critic_Score_Skew"] = all_df["Developer"].map(g[col].skew())

        ########################################################################################
        #出しているゲームの数
        #all_df["Dev_Game_Num"] = all_df["Developer"].map(vc)

        # 展開しているPlatformの数
        col = "Platform"
        #all_df["Dev_Plat_Num"] = all_df["Developer"].map(g[col].unique().map(lambda x : len(x)))
        return save_module(module, all_df, train_df.shape[0])
    else :
        return module


def Year_of_Release_encode(module, param):
    target_col = "Year_of_Release"
    if param["flag"]:
        train_df, test_df, all_df = pick_data(module)
        vc = all_df[target_col].value_counts()
        g = all_df.groupby(target_col)

        # 出している作品のうち、レビューがついているものの割合, 統計情報(User)
        col = "User_Count"
        #all_df["Year_of_Release_Reviewed_Rate"] = all_df[target_col].map(g[col].count() / vc)#
        #all_df["Release_of_Year_User_Count_Sum"] = all_df[target_col].map(g[col].sum())#
        #all_df["Release_of_Year_User_Count_Mean"] = all_df[target_col].map(g[col].mean())
        #all_df["Release_of_Year_User_Count_Median"] = all_df[target_col].map(g[col].median())
        #all_df["Release_of_Year_User_Count_Max"] = all_df[target_col].map(g[col].max())
        #all_df["Release_of_Year_User_Count_Min"] = all_df[target_col].map(g[col].min())
        #all_df["Release_of_Year_User_Count_Std"] = all_df[target_col].map(g[col].std())
        #all_df["Release_of_Year_User_Count_Skew"] = all_df[target_col].map(g[col].skew())

        # 出している作品のうち、レビューの平均(User)
        col = "User_Score"
        temp_g = all_df[all_df["User_Score"] != "tbd"]; temp_g["User_Score"] = temp_g["User_Score"].astype(float)
        temp_g = temp_g.groupby(target_col)
        all_df["Year_of_Release_User_Score_Sum"] = all_df[target_col].map(temp_g[col].sum())#
        #all_df["Year_of_Release_User_Score_Mean"] = all_df[target_col].map(temp_g[col].mean())
        #all_df["Year_of_Release_User_Score_Max"] = all_df[target_col].map(temp_g[col].max())
        #all_df["Year_of_Release_User_Score_Median"] = all_df[target_col].map(temp_g[col].median())
        #all_df["Year_of_Release_User_Score_Min"] = all_df[target_col].map(temp_g[col].min())
        #all_df["Year_of_Release_User_Score_Skew"] = all_df[target_col].map(temp_g[col].skew())

        # 出している作品のうち、レビューがついているものの割合, 統計情報(Critic)
        col = "Critic_Count"
        #all_df["Release_of_Year_Critic_Count_Sum"] = all_df[target_col].map(g[col].sum())
        #all_df["Release_of_Year_Critic_Count_Mean"] = all_df[target_col].map(g[col].mean())
        #all_df["Release_of_Year_Critic_Count_Median"] = all_df[target_col].map(g[col].median())
        #all_df["Release_of_Year_Critic_Count_Max"] = all_df[target_col].map(g[col].max())
        #all_df["Release_of_Year_Critic_Count_Min"] = all_df[target_col].map(g[col].min())
        #all_df["Release_of_Year_Critic_Count_Std"] = all_df[target_col].map(g[col].std())
        #all_df["Release_of_Year_Critic_Count_Skew"] = all_df[target_col].map(g[col].skew())

        # 出している作品のうち、レビューの統計情報(Critic))
        col = "Critic_Score"
        #all_df["Year_of_Release_Critic_Score_Sum"] = all_df[target_col].map(g[col].sum())
        #all_df["Year_of_Release_Critic_Score_Mean"] = all_df[target_col].map(g[col].mean())
        #all_df["Year_of_Release_Critic_Score_Median"] = all_df[target_col].map(g[col].median())
        #all_df["Year_of_Release_Critic_Score_Max"] = all_df[target_col].map(g[col].max())
        #all_df["Year_of_Release_Critic_Score_Min"] = all_df[target_col].map(g[col].min())
        #all_df["Year_of_Release_Critic_Score_Std"] = all_df[target_col].map(g[col].std())
        #all_df["Year_of_Release_Critic_Score_Skew"] = all_df[target_col].map(g[col].skew())

        #出しているゲームの数
        all_df["Year_of_Release_Game_Num"] = all_df[target_col].map(vc)

        # 展開しているPlatformの数
        col = "Platform"
        all_df["Year_of_Release_Plat_Num"] = all_df[target_col].map(g[col].unique().map(lambda x : len(x)))
        return save_module(module, all_df, train_df.shape[0])
    else :
        return module

def Rating_encode(module, param):
    target_col = "Rating"
    if param["flag"]:
        train_df, test_df, all_df = pick_data(module)
        vc = all_df[target_col].value_counts()
        g = all_df.groupby(target_col)

        # 出している作品のうち、レビューがついているものの割合, 統計情報(User)
        col = "User_Count"
        #all_df["Rating_Reviewed_Rate"] = all_df[target_col].map(g[col].count() / vc)
        #all_df["Rating_User_Count_Sum"] = all_df[target_col].map(g[col].sum())
        #all_df["Rating_User_Count_Mean"] = all_df[target_col].map(g[col].mean())
        #all_df["Rating_User_Count_Median"] = all_df[target_col].map(g[col].median())
        #all_df["Rating_User_Count_Max"] = all_df[target_col].map(g[col].max())
        #all_df["Rating_User_Count_Min"] = all_df[target_col].map(g[col].min())
        #all_df["Rating_User_Count_Std"] = all_df[target_col].map(g[col].std())
        #all_df["Rating_User_Count_Skew"] = all_df[target_col].map(g[col].skew())

        # 出している作品のうち、レビューの平均(User)
        col = "User_Score"
        temp_g = all_df[all_df["User_Score"] != "tbd"]; temp_g["User_Score"] = temp_g["User_Score"].astype(float)
        temp_g = temp_g.groupby(target_col)
        #all_df["Rating_User_Score_Sum"] = all_df[target_col].map(temp_g[col].sum())
        #all_df["Rating_User_Score_Mean"] = all_df[target_col].map(temp_g[col].mean())
        #all_df["Rating_User_Score_Max"] = all_df[target_col].map(temp_g[col].max())
        #all_df["Rating_User_Score_Median"] = all_df[target_col].map(temp_g[col].median())
        #all_df["Rating_User_Score_Min"] = all_df[target_col].map(temp_g[col].min())
        #all_df["Rating_User_Score_Skew"] = all_df[target_col].map(temp_g[col].skew())

        # 出している作品のうち、レビューがついているものの割合, 統計情報(Critic)
        col = "Critic_Count"
        #all_df["Rating_Critic_Count_Sum"] = all_df[target_col].map(g[col].sum())
        #all_df["Rating_Critic_Count_Mean"] = all_df[target_col].map(g[col].mean())
        #all_df["Rating_Critic_Count_Median"] = all_df[target_col].map(g[col].median())
        #all_df["Rating_Critic_Count_Max"] = all_df[target_col].map(g[col].max())
        #all_df["Rating_Critic_Count_Min"] = all_df[target_col].map(g[col].min())
        #all_df["Rating_Critic_Count_Std"] = all_df[target_col].map(g[col].std())
        #all_df["Rating_Critic_Count_Skew"] = all_df[target_col].map(g[col].skew())

        # 出している作品のうち、レビューの統計情報(Critic))
        col = "Critic_Score"
        #all_df["Rating_Critic_Score_Sum"] = all_df[target_col].map(g[col].sum())
        #all_df["Rating_Critic_Score_Mean"] = all_df[target_col].map(g[col].mean())
        #all_df["Rating_Critic_Score_Median"] = all_df[target_col].map(g[col].median())
        #all_df["Rating_Critic_Score_Max"] = all_df[target_col].map(g[col].max())
        #all_df["Rating_Critic_Score_Min"] = all_df[target_col].map(g[col].min())
        #all_df["Rating_Critic_Score_Std"] = all_df[target_col].map(g[col].std())
        #all_df["Rating_Critic_Score_Skew"] = all_df[target_col].map(g[col].skew())

        #出しているゲームの数
        #all_df["Rating_Game_Num"] = all_df[target_col].map(vc)

        # 展開しているPlatformの数
        #col = "Platform"
        #all_df["Rating_Plat_Num"] = all_df[target_col].map(g[col].unique().map(lambda x : len(x)))
        return save_module(module, all_df, train_df.shape[0])
    else :
        return module