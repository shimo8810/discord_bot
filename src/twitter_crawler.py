import datetime
from os import path
import json
import time
import tweepy
import unicodedata
import re
import csv

#PATH関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# deep learningディレクトリのrootパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

with open(path.join(ROOT_PATH, 'twitter_config.json'), 'r') as f:
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
    return text

class ReplyListener(tweepy.StreamListener):
    def __init__(self, api):
        super(ReplyListener, self).__init__()
        self.api = api

    def dump(self, in_txt, out_txt):
        with open(path.join(ROOT_PATH, 'twitter_corpus/corpus.csv'), 'a') as f:
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
        print('On Error:', status)

    def on_limit(self, status):
        print('On limit', status)

def main():
    auth = tweepy.OAuthHandler(CONFIG['ConsumerKey'], CONFIG['ConsumerSecret'])
    auth.set_access_token(CONFIG['AccessToken'], CONFIG['AccessTokenSecret'])
    api = tweepy.API(auth)

    listener = ReplyListener(api=api)
    stream = tweepy.Stream(auth=auth, listener=listener)

    try:
        while True:
            stream.sample(languages=["ja"])
    finally:
        stream.disconnect()
        print('finished proc')

if __name__ == '__main__':
    main()
