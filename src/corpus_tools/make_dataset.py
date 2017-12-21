"""
複合的な入出力sequenceファイルからvocabファイルを生成する
"""
import random
import argparse
from os import path
import subprocess
import MeCab
from tqdm import tqdm

def main():
    # 引数
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_seq', '-i', type=str, default='conversation_corpus/input_sequence.txt')
    parser.add_argument('--out_seq', '-o', type=str, default='conversation_corpus/output_sequence.txt')
    parser.add_argument('--out_dir', '-d', type=str, default='conversation_corpus/dataset')
    args = parser.parse_args()

    # MeCab準備
    mecab_tagger_option = '-Owakati -d '
    mecab_tagger_option += subprocess.check_output(['mecab-config', '--dicdir']).decode().strip()
    mecab_tagger_option += '/mecab-ipadic-neologd'

    tagger = MeCab.Tagger(mecab_tagger_option)

    # 単語辞書
    vocab_dict = {}
    # 学習データかテストデータか
    is_train = []
    print("#reading input sequence")
    # 入力文字列に対して
    with open(args.in_seq, 'r') as f:
        for line in tqdm(f):
            if random.random() * 100 >= 99:
                is_train.append(False)
            else:
                is_train.append(True)
            text = line.strip().split(' ')
            for word in text:
                if word in vocab_dict:
                    vocab_dict[word] += 1
                else:
                    vocab_dict[word] = 1

    print("#reading output sequence")
    with open(args.out_seq, 'r') as f:
        for line in tqdm(f):
            text = line.strip().split(' ')
            for word in text:
                if word in vocab_dict:
                    vocab_dict[word] += 1
                else:
                    vocab_dict[word] = 1

    print('# wtiring vocab files')
    with open('vocab.txt', 'w') as f1, \
         open('vocab_hist.txt', 'w') as f2, \
         open('vocab_denoise.txt', 'w') as f3:
        for k, v in tqdm(sorted(vocab_dict.items(), key=lambda x:x[1],reverse=True)):
            f2.writelines([k + '\t', str(v)+ '\n'])
            f1.write(k + '\n')
            if v > 2:
                f3.write(k + '\n')

if __name__  == '__main__':
    main()
