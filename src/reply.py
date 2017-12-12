from os import path
import json
import discord
import respond

#PATH関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# deep learningディレクトリのrootパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

# config file読み込み
with open(path.join(ROOT_PATH, 'bot_config.json'), 'r') as f:
    CONFIG = json.load(f)

#準備
client = discord.Client()
talker = respond.Talker(vocab_path=path.join(ROOT_PATH, CONFIG['VocabPath']),
                        model_path=path.join(ROOT_PATH, CONFIG['ModelPath']))

@client.event
async def on_message(message):
    # we do not want the bot to reply to itself
    if message.author == client.user:
        return

    print(message.channel)
    res = talker.response(message.content)
    print(message.content, " => ", res)
    await client.send_message(message.channel, res)
    # if message.content.startswith('!hello'):
    # msg = 'Hello {0.author.mention}'.format(message)
    # await client.send_message(message.channel, msg)

@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

if __name__ == '__main__':
    client.run(CONFIG['Token'])
