import argparse
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def get_log_urls_html_log(filepath):
    """
    Get logfile URLs from html log file
    Args:
        filepath(str): html log filepath (You can download from Tenho site.)
    Returns:
        log_urls(list): list of log urls
    """
    log_cols = ["start_time", "play_time", "game_type", "url", "player"]
    df = pd.read_csv(filepath, sep='|', names=log_cols)
    # 四麻に絞り込む
    df = df[df.game_type.str.startswith(' 四')]
    # logのダウンロード先URLを切り抜く
    log_urls = df["url"].str[10:-9]
    # ダウンロードできる形式にURLを変換（そのままだと観戦ページのリンクになっている）
    log_urls = log_urls.str.replace("\?log=", "log/?")
    # リスト形式に変換
    log_urls = log_urls.to_list()

    return log_urls


def download_one_log(url, save_path):
    with open(save_path, "wb") as f:
        res = requests.get(url)
        f.write(res.content)


def download_logs(args):
    # 天鳳からダウンロードしたzipファイルを解凍するとhtml.gzができるはず
    html_filepath_list = sorted(list(Path(args.target_dir).glob("*.html.gz")))
    if len(html_filepath_list) == 0:
        print("There is no html log files. Please check the filepath.")
        print(args.target_dir)
        return
    else:
        download_dir = Path(args.download_dir)
        download_dir.mkdir(exist_ok=True)
        for filepath in html_filepath_list:
            print(f"Download from {filepath}")
            download_date_dir = download_dir / str(filepath).split(".")[0]
            download_date_dir.mkdir(exist_ok=True)
            # get log urls from html file
            log_urls = get_log_urls_html_log(filepath)
            for log_url in tqdm(log_urls):
                save_path = str(download_date_dir / log_url.split("?")[1]) + ".mjlog"
                download_one_log(log_url, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target_dir",
        type=str,
        help="filepath of html log dir to download mj logs"
    )
    parser.add_argument(
        "-d", "--download_dir",
        default="../data/mj_log",
        type=str,
        help="download path of mj logfiles"
    )
    args = parser.parse_args()
    download_logs(args)
