import argparse
import xml
import pandas as pd
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm
import tenho_log
import mj_tehai


class MJGame():

    def _get_bakaze(self, round_str):
        if round_str[0] == "東":
            return 0
        elif round_str[0] == "南":
            return 1
        elif round_str[0] == "西":
            return 2
        elif round_str[0] == "北":
            return 3
        else:
            return -1

    def _get_kyoku_num(self, round_str):
        if len(round_str) == 2:
            return int(round_str[1])
        else:
            return int(round_str[1:3])

    def __init__(self, round_data):
        # ドラ情報
        self.dora_list = []
        # 場風
        self.bakaze = self._get_bakaze(round_data[0])
        # 何局目か
        self.kyoku_num = self._get_kyoku_num(round_data[0])
        # 本場
        self.honba = round_data[1]
        # リーチ供託棒
        self.n_kyoutaku_riichi = round_data[2]
        # 捨て牌情報
        self.sutehai_dict = {i: [] for i in range(4)}
        # リーチフラグ
        self.riichi_player = -1

    def set_parent(self, player):
        self.parent = player

    def set_haipai(self, hands_list):
        """
        配牌をセット(4人分)
        """
        self.tehai_dict = {}
        for i, hands in enumerate(hands_list):
            tehai = [0]*40
            for hai in hands:
                tehai[hai] += 1
            self.tehai_dict[i] = tehai.copy()

    def set_point(self, points):
        self.points = points

    def draw(self, event):
        """
        牌を引いた時
        """
        hai = event["tile"]
        player = event["player"]
        self.tehai_dict[player][hai] += 1
        assert(self.tehai_dict[player][hai] <= 4)

    def discard(self, event):
        """
        牌を捨てた時
        """
        hai = event["tile"]
        player = event["player"]
        self.tehai_dict[player][hai] -= 1
        assert(self.tehai_dict[player][hai] >= 0)
        # 捨て牌情報も更新
        self.sutehai_dict[player].append(event["tile"])

    def add_dora(self, event):
        self.dora_list.append(event["tile"])

    def set_riichi(self, event):
        assert(event["player"] in [0, 1, 2, 3])
        self.riichi_player = event["player"]

    def check_end(self):
        if self.riichi_player >= 0:
            return True
        else:
            return False

    def _get_agari_hai(self, tehai):
        agari_hai_list = []
        for i in range(40):
            if i % 10 == 0 or i > 37:
                continue
            if tehai[i] == 4:
                continue
            _tehai = tehai.copy()
            _tehai[i] += 1
            syanten = tehai_check.get_syanten(_tehai)
            if syanten == -1:
                agari_hai_list.append(i)
        return agari_hai_list

    def get_data(self):
        feats = {}
        # 場の情報
        feats["bakaze"] = self.bakaze
        feats["kyoku_num"] = self.kyoku_num
        feats["honba"] = self.honba
        feats["kyotaku"] = self.n_kyoutaku_riichi
        feats["dora"] = self.dora_list
        feats["parent"] = self.parent
        feats["riichi_player"] = self.riichi_player
        # あがり牌
        agari_hai_list = self._get_agari_hai(self.tehai_dict[self.riichi_player])
        feats["agari_hai"] = agari_hai_list
        # 手牌と捨て牌 (player0がリーチ人)
        feats["player0_tehai"] = self.tehai_dict[self.riichi_player]
        feats["player0_sutehai"] = self.sutehai_dict[self.riichi_player]
        feats["player0_point"] = self.points[self.riichi_player]
        feats["player1_tehai"] = self.tehai_dict[(self.riichi_player+1) % 4]
        feats["player1_sutehai"] = self.sutehai_dict[(self.riichi_player+1) % 4]
        feats["player1_point"] = self.points[(self.riichi_player+1) % 4]
        feats["player2_tehai"] = self.tehai_dict[(self.riichi_player+2) % 4]
        feats["player2_sutehai"] = self.sutehai_dict[(self.riichi_player+2) % 4]
        feats["player2_point"] = self.points[(self.riichi_player+2) % 4]
        feats["player3_tehai"] = self.tehai_dict[(self.riichi_player+3) % 4]
        feats["player3_sutehai"] = self.sutehai_dict[(self.riichi_player+3) % 4]
        feats["player3_point"] = self.points[(self.riichi_player+3) % 4]
        return feats


def get_mjlog_data(mjlog_path):
    data_list = []
    game = tenho_log.Game()
    game.decode(open(mjlog_path))
    for round_data in game.asdata()["rounds"]:
        # MJGameのクラスインスタンス生成
        mj_game = MJGame(round_data["round"])
        # 配牌の取得
        mj_game.set_haipai(round_data["hands"])
        # 親情報
        mj_game.set_parent(round_data["dealer"])
        # 点棒状況
        mj_game.set_point(round_data["points"])
        # eventデータの中身をループで取得
        for event in round_data["events"]:
            # つもるとき
            if event["type"] == "Draw":
                # 手牌を更新
                mj_game.draw(event)
            # 捨てる時
            elif event["type"] == "Discard":
                # 手牌を更新
                mj_game.discard(event)
                # リーチ後であればこの時点のデータを記録して終了する
                if mj_game.check_end():
                    data = mj_game.get_data()
                    data_list.append(data)
                    break
            # リーチ時
            elif event["type"] == "Riichi":
                mj_game.set_riichi(event)
            # ドラ追加時
            elif event["type"] == "Dora":
                mj_game.add_dora(event)
    return data_list


tehai_check = mj_tehai.MJTehai()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--target_dir",
        default="../data/mj_log",
        type=str,
        help="filepath of mjlog files for reading"
    )
    parser.add_argument(
        "-s", "--save_dir",
        default="../data/train_data",
        type=str,
        help="save path of train data"
    )
    args = parser.parse_args()

    data_list = []  # type: List[Dict]
    for mjlog_path in tqdm(list(Path(args.target_dir).glob("*/*mjlog"))):
        try:
            one_game_data = get_mjlog_data(mjlog_path)
        except:
            print("error")
            continue
        data_list += one_game_data
    # make dataframe
    data_df = pd.DataFrame(data_list)
    print(data_df.shape)
    print(data_df.tail())
    # save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    # format pickle because of saving list format
    save_path = save_dir / "train.pkl"
    data_df.to_pickle(save_path)
