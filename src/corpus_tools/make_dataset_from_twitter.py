"""
twitter 対話コーパス
"""
from os import path
import csv
import re
import unicodedata
from xml.sax.saxutils import unescape
import subprocess
from tqdm import tqdm
import MeCab
from glob import glob
#PATH関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# deep learningディレクトリのrootパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../../'))

mecab_tagger_option = '-Owakati -d '
mecab_tagger_option += subprocess.check_output(['mecab-config', '--dicdir']).decode().strip()
mecab_tagger_option += '/mecab-ipadic-neologd'

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

def main():
    # twitter ユーザ名 正規表現
    twitter_name_pattern = re.compile(r'@[a-zA-Z0-9_]{1,32}')

    seq_pairs = []
    corpus_paths = glob(path.join(ROOT_PATH, 'conversation_corpus/twitter_corpus/*.csv'))
    for corpus_path in corpus_paths:
        print("# Reading {}".format(corpus_path.split('/')[-1]))
        with open(corpus_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):

                # 対話文章が失敗している対は無視
                if len(row) != 2:
                    continue
                in_seq, out_seq = row
                in_seq = twitter_name_pattern.sub('', in_seq.strip())
                out_seq = twitter_name_pattern.sub('', out_seq.strip())
                seq_pairs.append((in_seq, out_seq))

    tagger = MeCab.Tagger(mecab_tagger_option)
    with  open(path.join(ROOT_PATH, 'conversation_corpus/twitter_corpus/input_sequence_twitter.txt'), 'w') as f_in, \
          open(path.join(ROOT_PATH, 'conversation_corpus/twitter_corpus/output_sequence_twitter.txt'), 'w') as f_out:

        for in_seq, out_seq in tqdm(seq_pairs):
             # 入力側
            in_seq = tagger.parse(in_seq)
            f_in.write(in_seq)

            # 出力側
            out_seq = tagger.parse(out_seq)
            f_out.write(out_seq)

if __name__ == '__main__':
    main()
