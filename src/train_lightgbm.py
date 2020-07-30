import pickle
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import log_loss


def func_33_to_40(x):
    if x < 9:
        return x+1
    elif x < 18:
        return x+2
    elif x < 27:
        return x+3
    else:
        return x+4


def make_feature1(df):
    # ドラの数
    df["dora_num"] = df["dora"].map(len)
    # 捨て牌の数
    df["sutehai_num"] = df["player0_sutehai"].map(len)
    # 捨て牌の字牌率
    df["sutehai_jihai_rate"] = df["player0_sutehai"].map(lambda h_list: np.mean([h > 30 for h in h_list]))
    # 捨て牌の19字牌率
    df["sutehai_yaochu_rate"] = df["player0_sutehai"].map(
        lambda h_list: np.mean([h > 30 or h%10 ==1 or h%10 ==9 for h in h_list]))
    # 捨て牌の456牌率
    df["sutehai_456_rate"] = df["player0_sutehai"].map(
        lambda h_list: np.mean([h%10 in [4, 5, 6] for h in h_list]))
    # 宣言牌
    sengen_hai = df["player0_sutehai"].map(lambda x: x[-1])
    # 宣言牌の数字（字牌は-1）
    df["sengen_hai"] = sengen_hai
    df["sengen_hai_number"] = sengen_hai.map(lambda x: x%10 if x < 30 else -1)
    df["sengen_hai_color"] = sengen_hai.map(lambda x: x//10)
    df["sengen_hai_count"] = sengen_hai.map(sengen_hai.value_counts(normalize=True))
    return df


def make_feature2(df):
    def contain_dora(df):
        # 最初のドラが捨て牌に含まれているか
        df["idx"] = df.index
        dora_dict = df["dora"].map(lambda x: x[0]).to_dict()
        sutehai_dict = df["player0_sutehai"].to_dict()
        def contain_func(idx):
            dora = dora_dict[idx]
            sutehai = sutehai_dict[idx]
            return int(dora in set(sutehai))
        dora_in_sutehai = df["idx"].map(contain_func)
        del df["idx"]
        return dora_in_sutehai
    
    # 捨て牌にドラ(1つ目)が含まれているか(0 or 1)
    df["sutehai_dora_sum"] = contain_dora(df)
    return df


def make_feature3(df):
    # 序盤(~3, ~6)にヤオチュウ牌以外を切っている割合
    df["ratio_not_yaochu_in_first3"] = df["player0_sutehai"].map(
        lambda h_list: np.mean([h < 30 and h%10 in (2, 3, 4, 5, 6, 7, 8) for h in h_list[:3]]))
    df["ratio_not_yaochu_in_first6"] = df["player0_sutehai"].map(
        lambda h_list: np.mean([h < 30 and h%10 in (2, 3, 4, 5, 6, 7, 8) for h in h_list[:6]]))
    # 序盤(~3, ~6)に4, 5, 6を切っている割合
    df["ratio_456_in_first3"] = df["player0_sutehai"].map(
        lambda h_list: np.mean([h < 30 and h%10 in (4, 5, 6) for h in h_list[:3]]))
    df["ratio_456_in_first6"] = df["player0_sutehai"].map(
        lambda h_list: np.mean([h < 30 and h%10 in (4, 5,  6) for h in h_list[:6]]))
    return df


def make_feature4(df):
    # 捨て牌で各牌を何枚ずつ切ったか
    sutehai_count = df["player0_sutehai"].apply(lambda x: pd.Series(np.eye(40)[x].sum(axis=0)))
    use_col = [c for c in range(40) if c < 38 and c%10 != 0]
    sutehai_count = sutehai_count.loc[:, use_col]
    sutehai_count.columns = [f"discard_{c}" for c in use_col]
    df = pd.concat([df, sutehai_count], axis=1)
    return df


def make_feature5(df):
    # 各牌が場に何枚見えているか
    # 各プレイヤーの捨て牌
    sutehai_count0 = df["player0_sutehai"].apply(lambda x: pd.Series(np.eye(40)[x].sum(axis=0)))
    sutehai_count1 = df["player1_sutehai"].apply(lambda x: pd.Series(np.eye(40)[x].sum(axis=0)))
    sutehai_count2 = df["player2_sutehai"].apply(lambda x: pd.Series(np.eye(40)[x].sum(axis=0)))
    sutehai_count3 = df["player3_sutehai"].apply(lambda x: pd.Series(np.eye(40)[x].sum(axis=0)))
    sutehai_count_all = sutehai_count0 + sutehai_count1 + sutehai_count2 + sutehai_count3
    # プレイヤー1目線として、プレイヤー1の手牌
    tehai_count1 = df["player1_tehai"].apply(lambda x: pd.Series(np.eye(40)[x].sum(axis=0)))
    # ドラ表示牌
    def pre_dora_func(x):
        if x < 30:
            if x%10 > 1:
                return x-1
            else:
                return x//10 + 9
        else:
            if x == 31:
                return 34
            if x == 35:
                return 37
            else:
                return x-1

    pre_dora = df["dora"].map(
        lambda h_list: [pre_dora_func(h) for h in h_list]).apply(lambda x: pd.Series(np.eye(40)[x].sum(axis=0)))
    # sum
    count_all = sutehai_count_all + tehai_count1
    use_col = [c for c in range(40) if c < 38 and c%10 != 0]
    count_all = count_all.loc[:, use_col]
    count_all.columns = [f"can_see_count_{c}" for c in use_col]
    df = pd.concat([df, count_all], axis=1)
    return df


# add feature
def make_feature_color(df):
    sutehai_num = df["player0_sutehai"].map(len)
    df["manzu_ratio"] = df["player0_sutehai"].map(lambda h_list:
        sum([h for h in h_list if h < 10])
    ) / sutehai_num
    df["pinzu_ratio"] = df["player0_sutehai"].map(lambda h_list:
        sum([h for h in h_list if h > 10 and h < 20])
    ) / sutehai_num
    df["souzu_ratio"] = df["player0_sutehai"].map(lambda h_list:
        sum([h for h in h_list if h > 20 and h < 30])
    ) / sutehai_num
    df["jihai_ratio"] = df["player0_sutehai"].map(lambda h_list:
        sum([h for h in h_list if h > 30])
    ) / sutehai_num
    # 種類毎に最初に切った牌
    def func_first_hai(hai_list, hai_type):
        if hai_type == "manzu":
            target_list = [h for h in hai_list if h < 10]
        elif hai_type == "pinzu":
            target_list = [h for h in hai_list if h > 10 and h < 20]
        elif hai_type == "souzu":
            target_list = [h for h in hai_list if h > 20 and h < 30]
        else:
            target_list = [h for h in hai_list if h > 30]
        if len(target_list) > 0:
            return target_list[0]
        else:
            return -1
    df["manzu_first_hai"] = df["player0_sutehai"].map(
        lambda h_list: func_first_hai(h_list, "manzu")
    )
    df["pinzu_first_hai"] = df["player0_sutehai"].map(
        lambda h_list: func_first_hai(h_list, "pinzu")
    )
    df["souzu_first_hai"] = df["player0_sutehai"].map(
        lambda h_list: func_first_hai(h_list, "souzu")
    )
    df["jihai_first_hai"] = df["player0_sutehai"].map(
        lambda h_list: func_first_hai(h_list, "jihai")
    )
    return df


def make_ohe_target(y_srs):
    y_ohe = y_srs.apply(lambda x: pd.Series(np.eye(40)[x].sum(axis=0)))
    use_col = [c for c in range(40) if c < 38 and c%10 != 0]
    y_ohe = y_ohe.loc[:, use_col]
    return y_ohe


def lgb_train(df, ohe_targets):
    except_feats = [
        "dora",
        "agari_hai",
        "player0_tehai",
        "player1_tehai",
        "player2_tehai",
        "player3_tehai",
        "player0_sutehai",
        "player1_sutehai",
        "player2_sutehai",
        "player3_sutehai",
    ]
    use_feats = [c for c in df.columns if c not in except_feats]
    
    lgb_params = {
            'objective':'binary',
            "metric":"binary_logloss",
            "verbosity": -1,
            "boosting": "gbdt",
            'learning_rate': 0.1,
            'num_leaves': 32,
            'min_data_in_leaf': 30, 
            'max_depth': 4,
            "bagging_freq": 1,
            "bagging_fraction": 0.7,
            "bagging_seed": 11,
            "lambda_l1": 0.3,
            "lambda_l2": 0.3,
            "feature_fraction": 0.7,
            "seed": 11,
            "num_threads": 4,
    }
    importances = pd.DataFrame()
    models = {}

    folds = KFold(n_splits=5, shuffle=True, random_state=2020)
    train_X, valid_X, train_y,valid_y = train_test_split(
        df, ohe_targets, test_size=0.2, shuffle=True, random_state=2020
    )
    val_preds = np.zeros(valid_y.shape)

    train_X = train_X[use_feats].values
    train_y = train_y.values
    valid_X = valid_X[use_feats].values
    valid_y = valid_y.values

    for hai_idx, hai_col in enumerate(ohe_targets.columns):
        print("="*30)
        print(hai_col)
        print("="*30)
        trn_data = lgb.Dataset(
            train_X,
            label=train_y[:, hai_idx]
        )
        val_data = lgb.Dataset(
            valid_X,
            label=valid_y[:, hai_idx]
        )

        model = lgb.train(
                    lgb_params,
                    trn_data,
                    5000,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=1000,
                    early_stopping_rounds = 200,
        )
        imp_df = pd.DataFrame()
        imp_df['feature'] = use_feats
        imp_df['gain'] = model.feature_importance(importance_type="gain")
        imp_df["hai"] = hai_col
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

        val_preds[:, hai_idx] = model.predict(valid_X)
        models[hai_col] = model

    return val_preds, importances, models, use_feats


def exists_train_data():
    return Path("../data/df.pkl").exists() and Path("../data/ohe_targets.pkl").exists()


def train_run():
   
    if not exists_train_data():
        print("load data")
        df = pd.read_pickle("../data/train_data/train.pkl")
        if DEBUG:
            df = df.sample(1000).reset_index(drop=True)
        print("dora map")
        df["dora"] = df["dora"].map(lambda h_list: [func_33_to_40(h//4) for h in h_list])
        # メモリ削減のためplayer1以外の手牌のカラムは消す
        df.drop(["player0_tehai", "player2_tehai", "player3_tehai"], axis=1, inplace=True)
        # Make feature
        print("make feature1")
        df = make_feature1(df)
        print("make feature2")
        df = make_feature2(df)
        print("make feature3")
        df = make_feature3(df)
        print("make feature4")
        df = make_feature4(df)
        print("make feature5")
        df = make_feature5(df)
        # select only agaqri-hai exists
        df = df[df.agari_hai.map(len) > 0].reset_index(drop=True)
        # make one-hot target
        print("make ohe targets")
        ohe_targets = make_ohe_target(df["agari_hai"])
        # save data
        print("save data")
        df.to_pickle("../data/df.pkl")
        ohe_targets.to_pickle("../data/ohe_targets.pkl")
    else:
        df = pd.read_pickle("../data/df.pkl")
        ohe_targets = pd.read_pickle("../data/ohe_targets.pkl")

    # lightgbm training
    val_preds, importances, models, use_feats = lgb_train(df, ohe_targets)
    # save data
    with open("../data/val_preds.pkl", "wb") as f:
        pickle.dump(val_preds, f)
    with open("../data/importances.pkl", "wb") as f:
        pickle.dump(importances, f)
    with open("../data/models.pkl", "wb") as f:
        pickle.dump(models, f)
    with open("../data/use_feats.pkl", "wb") as f:
        pickle.dump(use_feats, f)



if __name__ == "__main__":
    DEBUG = True
    train_run()
