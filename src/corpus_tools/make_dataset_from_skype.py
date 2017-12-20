from os import path
import csv
import mojimoji
from tqdm import tqdm
import MeCab
import re
import unicodedata
from xml.sax.saxutils import unescape
import subprocess

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

if __name__ == '__main__':
    tagger = MeCab.Tagger(mecab_tagger_option)

    sequence_pairs = []
    seq_in = None
    seq_out = None
    speaker_in = None
    speaker_out = None
    sequence_pairs = []
    with open(path.join(ROOT_PATH, 'conversation_corpus/skype_corpus/skype_conv_log.csv'), 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if speaker_in is None:
                speaker_in, seq_in = row
                continue
            # 今の発話者が質問者と異なる場合
            if row[0] != speaker_in:
                speaker_out, seq_out = row

                # 全角を半角に変換
                si = normalzie(seq_in)
                so = normalzie(seq_out)
                # sequenceペアに追加
                sequence_pairs.append((si, so))

                speaker_in, seq_in = speaker_out, seq_out
            else:
                speaker_in, seq_in = row

    with  open(path.join(ROOT_PATH, 'conversation_corpus/skype_corpus/input_sequence_skype.txt'), 'w') as f_in, \
          open(path.join(ROOT_PATH, 'conversation_corpus/skype_corpus/output_sequence_skype.txt'), 'w') as f_out:

        for seq_in, seq_out in tqdm(sequence_pairs):
            # 入力側
            seq_in = tagger.parse(seq_in)
            f_in.write(seq_in)
            seq_in = seq_in.split(' ')

            # 出力側
            seq_out = tagger.parse(seq_out)
            f_out.write(seq_out)
            seq_out = seq_out.split(' ')
