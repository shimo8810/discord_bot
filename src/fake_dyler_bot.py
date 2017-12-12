from os import path
import json
import argparse
import asyncio
import discord
# 自作返答ライブラリ
import respond
import random

#PATH関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# deep learningディレクトリのrootパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

class FakeDylerBot(discord.Client):
    def __init__(self, vocab_path, model_path):
        super(FakeDylerBot, self).__init__()
        self.is_online = None
        self.talker = respond.Talker(vocab_path=path.join(ROOT_PATH, vocab_path),
                                     model_path=path.join(ROOT_PATH, model_path))

    async def on_ready(self):
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')
        await self.change_presence(status=discord.Status.online)
        self.is_online = True

    async def on_message(self, message):
        if message.author == self.user:
            return

        # 起動
        if message.content.startswith('$起きて'):
            await self.change_presence(status=discord.Status.online)
            msg = "何?"
            await self.send_message(message.channel, msg)
            self.is_online = True
            return

        # 終了
        elif message.content.startswith('$おやすみ'):
            if not self.is_online:
                msg = 'Zzz...'
            else:
                msg = 'もう寝るわ'
            await self.send_message(message.channel, msg)
            await self.change_presence(status=discord.Status.dnd)
            self.is_online = False
            return

        # 通常のreply
        else:
            if self.is_online:
                res = self.talker.response(message.content)
                print(message.channel, ":", message.content, " => ", res)
                await self.send_message(message.channel, res)

def main():
    # 引数
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', '-v', type=str, default='seq2seq_model/vocab_skype_nucc.txt')
    parser.add_argument('--model', '-m', type=str, default='seq2seq_model/seq2seq_conversation.npz')
    # parser.add_argument('--gpu', '-g', type=int, default=-1)
    # parser.add_argument('--layer', '-l', type=int, default=3)
    # parser.add_argument('--unit', '-u', type=int, default=256)
    parser.add_argument('--dev', action='store_true', default=False)
    args = parser.parse_args()

    # クライアント準備
    client = FakeDylerBot(vocab_path=path.join(ROOT_PATH, args.vocab),
                          model_path=path.join(ROOT_PATH, args.model))

    # configファイル読み込み
    config_file = 'testbot_config.json' if args.dev else 'bot_config.json'
    with open(path.join(ROOT_PATH, config_file), 'r') as f:
        conf = json.load(f)
    client.run(conf['Token'])

if __name__ == '__main__':
    main()
