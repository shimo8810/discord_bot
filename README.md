# Discord用対話BOT

## tl;dr
僕がいなくてもいいように僕の会話ログを学習したbot

## ToDo
- [ ] 返答が必要なときだけ返答するようにする
- [ ] channel毎に状態を決定する
- [ ] コーパスのクレンジング
    - [ ] 名大会話コーパス固有の文字列を削除
    - [ ] Skypeログの絵文字やAA,コードを除外
    - [ ] 数値を除外する
- [ ] ネットワークに関する改善
    - [ ] 最初からword2vecで分散表現を学習しておくとか?
    - [ ] attention netに改変

## 使用対話コーパス
- [名大会話コーパス](http://mmsrv.ninjal.ac.jp/nucc/)
- Skype会話ログ
- Twitter(予定)
