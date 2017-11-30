
import os
import math
from pprint import pprint

from collections import Counter
import argparse

from sklearn import metrics
import torch
import torch.nn as nn

from seqmod.misc.preprocess import text_processor
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.trainer import Trainer
from seqmod.misc.loggers import StdLogger
from seqmod.misc.dataset import PairedDataset
from seqmod.misc.early_stopping import EarlyStopping
from seqmod.hyper import make_sampler, Hyperband
import seqmod.utils as u

from text_nn.loaders import load_twisty, load_dataset, load_embeddings

from text_nn import models


def compute_scores(model, dataset):
    trues, preds = [], []
    for sources, targets in dataset:
        _, b_preds = model.predict(sources)
        trues.extend(targets.view(-1).data.tolist())
        preds.extend(b_preds.view(-1).data.tolist())
    return trues, preds


def make_score_hook(model, dataset):

    def hook(trainer, epoch, batch, checkpoint):
        trues, preds = compute_scores(model, dataset)
        trainer.log("info", metrics.classification_report(trues, preds))

    return hook


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--emb_dim', default=50, type=int)
    parser.add_argument('--hid_dim', default=50, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--load_embeddings', action='store_true')
    parser.add_argument('--flavor', default=None)
    parser.add_argument('--suffix', default=None)
    parser.add_argument('--max_dim', default=100, type=int)
    parser.add_argument('--out_channels', default=(12,), nargs='+', type=int)
    parser.add_argument('--kernel_sizes', nargs='+', type=int,
                        default=(5, 4, 3))
    parser.add_argument('--act', default='relu')
    parser.add_argument('--ktop', default=4, type=int)
    # training
    parser.add_argument('--optim', default='Adam')
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--max_norm', default=20., type=float)
    parser.add_argument('--batch_size', type=int, default=264)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--checkpoints', default=100, type=int)
    parser.add_argument('--hooks_per_epoch', default=10, type=int)
    parser.add_argument('--max_iter', default=81, type=int)
    parser.add_argument('--eta', default=3, type=int)
    # dataset
    parser.add_argument('--dev', default=0.1, type=float)
    parser.add_argument('--test', default=0.2, type=float)
    parser.add_argument('--min_len', default=5, type=int)
    parser.add_argument('--min_freq', default=5, type=int)
    parser.add_argument('--level', default='token')
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--cache_data', action='store_true')
    parser.add_argument('--max_tweets', type=int, default=0)
    args = parser.parse_args()

    print("Loading data...")
    prefix = '{level}.{min_len}.{min_freq}.{concat}.{max_tweets}'.format(**vars(args))
    if not args.cache_data or not os.path.isfile('data/{}_train.pt'.format(prefix)):
        src, trg = load_twisty(
            min_len=args.min_len, level=args.level, concat=args.concat,
            processor=text_processor(lower=False),
            max_tweets=None if args.max_tweets == 0 else args.max_tweets)
        train, test, valid = load_dataset(
            src, trg, args.batch_size, min_freq=args.min_freq,
            gpu=args.gpu, dev=args.dev, test=args.test)
        if args.cache_data:
            train.to_disk('data/{}_train.pt'.format(prefix))
            test.to_disk('data/{}_test.pt'.format(prefix))
            valid.to_disk('data/{}_valid.pt'.format(prefix))
    else:
        train = PairedDataset.from_disk('data/{}_train.pt'.format(prefix))
        valid = PairedDataset.from_disk('data/{}_valid.pt'.format(prefix))
    train.set_gpu(args.gpu), valid.set_gpu(args.gpu)
    datasets = {'train': train, 'valid': valid}

    weight = None
    if args.load_embeddings:
        print("Loading pretrained embeddings")
        weight = load_embeddings(
            train.d['src'].vocab, args.flavor, args.suffix, 'data')

    print("Starting experiment")
    sampler = make_sampler({
        'emb_dim': ['uniform', int, 20, 100],
        'hid_dim': ['uniform', int, 20, 100],
        'dropout': ['loguniform', float, math.log(0.1), math.log(0.5)],
        'model': ['choice', str, ('CNNText', 'DCNN', 'RCNN', 'RNNText', 'ConvRec')],
        # 'load_embeddings': ['choice', bool, (True, False)],
        'max_dim': ['uniform', int, 50, 200],
        # not applying to DCNN
        'out_channels': ['uniform', int, 10, 150],
        'kernel_sizes': ['choice', tuple, [
            (5, 4, 3), (7, 5, 4, 3), (9, 7, 5, 4, 3),
            (7, 5, 3), (9, 5, 3, 2), (12, 9, 6, 3)]],
        # only applying to DCNN: increase kernel_sizes, out_channels by factor
        'dcnn_factor': ['uniform', int, 1, 3], 'ktop': ['uniform', int, 3, 8],
        # 'lr': ['loguniform', float, math.log(0.001), math.log(0.05)]
    })

    vocab, n_classes = len(train.d['src'].vocab), len(train.d['trg'].vocab)

    class create_runner(object):
        def __init__(self, params):
            self.trainer, self.early_stopping = None, None

            if params['model'] == 'DCNN':
                kernel_sizes, out_channels = (7, 5), (6, 14),
                out_channels = tuple(
                    [c * params['dcnn_factor'] for c in out_channels])
                kernel_sizes = tuple(
                    [k * params['dcnn_factor'] for k in kernel_sizes])
            else:
                kernel_sizes = params['kernel_sizes']
                out_channels = params['out_channels']

            if params.get('load_embeddings', None):
                if not args.load_embeddings:
                    raise ValueError("Need load_embeddings")
                emb_dim = args.emb_dim
            else:
                emb_dim = params['emb_dim']

            model = getattr(models, params['model'])(
                n_classes, vocab, emb_dim=emb_dim, hid_dim=params['hid_dim'],
                dropout=params['dropout'],
                padding_idx=train.d['src'].get_pad(),
                # cnn
                act=args.act, out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                # - rcnn only
                max_dim=params['max_dim'],
                # - DCNN only
                ktop=params['ktop'])

            u.initialize_model(model)

            if params.get('load_embeddings', None):
                model.init_embeddings(weight)

            optimizer = Optimizer(
                model.parameters(), args.optim, lr=args.learning_rate,
                max_norm=args.max_norm, weight_decay=args.weight_decay)

            self.early_stopping = EarlyStopping(
                5, patience=3, reset_patience=False)

            def early_stop_hook(trainer, epoch, batch_num, num_checkpoints):
                valid_loss = trainer.validate_model()
                self.early_stopping.add_checkpoint(sum(valid_loss.pack()))

            trainer = Trainer(model, datasets, optimizer)
            # trainer.add_loggers(StdLogger())
            # trainer.add_hook(make_score_hook(model, valid),
            #                  hooks_per_epoch=args.hooks_per_epoch)
            trainer.add_hook(early_stop_hook, hooks_per_epoch=5)

            self.trainer = trainer

        def __call__(self, n_iters):
            batches = int(len(train) / args.max_iter) * 5
            print("Training {}".format(batches * n_iters))

            if args.gpu:
                self.trainer.model.cuda()

            (_, loss), _ = self.trainer.train_batches(
                batches * n_iters, args.checkpoints, shuffle=True)

            self.trainer.model.cpu()

            return {'loss': loss, 'early_stop': self.early_stopping.stopped}

    hb = Hyperband(
        sampler, create_runner, max_iter=args.max_iter, eta=args.eta)

    result = hb.run()

    pprint(result)
