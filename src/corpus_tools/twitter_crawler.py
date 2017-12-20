import datetime
from os import path
import json
import time
import tweepy
import unicodedata
import re
import csv
from xml.sax.saxutils import unescape
import urllib3

#PATH関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# deep learningディレクトリのrootパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../../'))

with open(path.join(ROOT_PATH, 'config/twitter_config.json'), 'r') as f:
    CONFIG = json.load(f)

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def normalzie(text):
    # unicode正規化
    text = unicodedata.normalize('NFKC', text)
    # emoji 除去
    text = emoji_pattern.sub('', text)
    # 改行タブ文字除去
    text = text.replace('\n', ' ').replace('\t', ' ')
    # htmlエスケープ文字の修正
    return unescape(text)

class ReplyListener(tweepy.StreamListener):
    def __init__(self, api, file_name):
        super(ReplyListener, self).__init__()
        self.api = api
        self.file_name = file_name

    def dump(self, in_txt, out_txt):
        with open(path.join(ROOT_PATH, 'conversation_corpus/twitter_corpus/{}_corpus.csv'.format(self.file_name)), 'a') as f:
            writer = csv.writer(f, lineterminator='\n', delimiter='\t')
            writer.writerow((in_txt, out_txt))

    def on_status(self, status):
        # tweetがreplyなら
        if status.in_reply_to_status_id:
            # URLは抜く
            if 'http' not in status.text:
                rep_text = normalzie(status.text)
                try:
                    to_stat = self.api.get_status(status.in_reply_to_status_id)
                except tweepy.error.TweepError as err:
                    print(err)
                    return False
                else:
                    if 'http' not in to_stat.text:
                        twt_text = normalzie(to_stat.text)
                        print(twt_text, '<-', rep_text)
                        self.dump(twt_text, rep_text)
        return True

    def on_error(self, status):
        time.sleep(60)
        print('On Error:', status)

    def on_limit(self, status):
        time.sleep(60)
        print('On limit', status)

def main():
    # 認証
    auth = tweepy.OAuthHandler(CONFIG['ConsumerKey'], CONFIG['ConsumerSecret'])
    auth.set_access_token(CONFIG['AccessToken'], CONFIG['AccessTokenSecret'])
    api = tweepy.API(auth)

    # ファイル名 (今日の日付)
    file_name = datetime.date.today().strftime('%Y%m%d')

    # stream 準備
    listener = ReplyListener(api=api, file_name=file_name)
    stream = tweepy.Stream(auth=auth, listener=listener)

    # がばがばerror handling
    try:
        while True:
            try:
                stream.sample(languages=["ja"])
            except urllib3.exceptions.ProtocolError as err:
                print(err)
                time.sleep(300)
    finally:
        stream.disconnect()
        print('finished proc')

if __name__ == '__main__':
    main()
