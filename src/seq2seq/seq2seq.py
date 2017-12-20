import os
from os import path
import argparse

import platform

if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

#PATH関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# deep learningディレクトリのrootパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../../'))

UNK = 0
EOS = 1

def sequence_embed(embed, xs):
    # embedにまとめて入れるために区切りを保存する
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    # まとめて入力するためにconcat
    ex = embed(F.concat(xs, axis=0))
    # データを再分割
    exs = F.split_axis(ex, x_section, 0)
    return exs

class Seq2seq(chainer.Chain):
    """
    seq2seqモデル
    """
    def __init__(self, n_layers, n_vocab, n_units):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.W = L.Linear(n_units, n_vocab)

        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, xs, ys):
        # xsを逆順に
        xs = [x[::-1] for x in xs]
        # ysにeosを挟む
        eos = self.xp.array([EOS], 'i')
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        #
        exs = sequence_embed(self.embed, xs)
        eys = sequence_embed(self.embed, ys_in)

        batch = len(xs)

        hx, cx, _ = self.encoder(None, None, exs)
        _, _, os = self.decoder(hx, cx, eys)

        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(self.W(concat_os), concat_ys_out, reduce='no')) / batch
        chainer.report({'loss': loss.data}, self)
        return loss

def load_vocab(vocab_path, ratio=1.0):
    """
    語彙idを返す関数
    '***' は<UNK>と統一で良さげ?
    """
    with open(path.join(ROOT_PATH, vocab_path), 'r') as f:
        word_ids = {line.strip() : i + 2 for i, line in enumerate(f)}
    # word_ids = {}
    # with open(vocab_path, 'r') as f:
    #     length = len(list(f)) * ratio
    #     for i, line in enumerate(f):
    #         if i + 2 > length:
    #             break
    #         word_ids[line.strip()] = i + 2
    word_ids['<UNK>'] = UNK
    word_ids['<EOS>'] = EOS
    return word_ids

def load_data(vocab, seq_in, seq_out):
    """
    データセットを返す関数
    """
    x_data = []
    y_data = []
    with open(path.join(ROOT_PATH, seq_in), 'r') as f:
        for line in f:
            words = line.strip().split(' ')
            x_data.append(np.array([vocab.get(w, UNK) for w in words], 'i'))
    with open(path.join(ROOT_PATH, seq_out), 'r') as f:
        for line in f:
            words = line.strip().split(' ')
            y_data.append(np.array([vocab.get(w, UNK) for w in words], 'i'))

    if len(x_data) != len(y_data):
        raise ValueError('len(x_data) != len(y_data)')

    data = [(x, y) for x, y in zip(x_data, y_data)]

    return data

def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif  device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}

def main():
    """
    main関数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', '-v', type=str, default='conversation_corpus/vocab.txt')
    parser.add_argument('--seq_in', '-i', type=str, default='conversation_corpus/input_sequence.txt')
    parser.add_argument('--seq_out', '-o', type=str, default='conversation_corpus/output_sequence.txt')
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--log_epoch', type=int, default=1)
    parser.add_argument('--alpha', '-a', type=float, default=0.001)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--batch', '-b', type=int, default=64)
    parser.add_argument('--layer', '-l', type=int, default=3)
    parser.add_argument('--unit', '-u', type=int, default=256)
    parser.add_argument('--vocab_ratio', '-r', type=float, default=1.0)
    parser.add_argument('--lr_shift', '-s', action='store_true', default=False)
    args = parser.parse_args()

    # save didrectory
    outdir = path.join(ROOT_PATH, 'seq2seq_results/seq2seq_conversation_epoch_{}_layer_{}_unit_{}_vr_{}'.format(
        args.epoch, args.layer, args.unit, args.vocab_ratio))
    if not path.exists(outdir):
        os.makedirs(outdir)
    with open(path.join(outdir, 'arg_param.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}:{}\n'.format(k, v))

    # print param
    print('# GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batch))
    print('# Epoch: {}'.format(args.epoch))
    print('# Adam alpha: {}'.format(args.alpha))
    print('# embedID unit :{}'.format(args.unit))
    print('# LSTM layer :{}'.format(args.layer))
    print('# out directory :{}'.format(outdir))
    print('# lr shift: {}'.format(args.lr_shift))
    print('')

    # load dataset
    vocab_ids = load_vocab(args.vocab, 0.7)
    train_data = load_data(vocab_ids, args.seq_in, args.seq_out)

    # prepare model
    model = Seq2seq(n_layers=args.layer, n_vocab=len(vocab_ids), n_units=args.unit)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # optimizer
    optimizer = chainer.optimizers.Adam(alpha=args.alpha)
    optimizer.setup(model)

    # iter
    train_iter = chainer.iterators.SerialIterator(train_data, args.batch)
    # trainer
    updater = training.StandardUpdater(train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=outdir)

    # extention
    # lr shift
    if args.lr_shift:
        trainer.extend(extensions.ExponentialShift("alpha", 0.1), trigger=(200, 'epoch'))
    # log
    trainer.extend(extensions.LogReport(trigger=(args.log_epoch, 'epoch')))
    trainer.extend(extensions.observe_lr(), trigger=(args.log_epoch, 'epoch'))
    # print info
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'lr', 'elapsed_time']), trigger=(args.log_epoch, 'epoch'))
    # print progbar
    trainer.extend(extensions.ProgressBar())
    # plot loss graph
    trainer.extend(
        extensions.PlotReport(['main/loss'], 'epoch', file_name='loss.png'))
    # save snapshot and model
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'model_snapshot_{.updater.epoch}'), trigger=(10, 'epoch'))

    # start learn
    print('start training')
    trainer.run()

    # save final model
    chainer.serializers.save_npz(path.join(outdir, "seq2seq_conversation_model.npz"), model)

if __name__ == '__main__':
    main()
