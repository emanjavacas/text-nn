
import os
import math

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

from loaders import load_twisty, load_dataset, load_embeddings

import models


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


def make_criterion(train):
    counts = Counter([y[0] for y in train.data['trg']])
    total = sum(counts.values())
    weight = torch.Tensor(len(counts)).zero_()
    for label, count in counts.items():
        weight[label] = total / count
    return nn.NLLLoss(weight=weight)


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
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--checkpoints', default=100, type=int)
    parser.add_argument('--hooks_per_epoch', default=10, type=int)
    parser.add_argument('--max_iter', default=81, type=int)
    parser.add_argument('--eta', default=3, type=int)
    # dataset
    parser.add_argument('--dev', default=0.1, type=float)
    parser.add_argument('--test', default=0.2, type=float)
    parser.add_argument('--min_len', default=0, type=int)
    parser.add_argument('--min_freq', default=5, type=int)
    parser.add_argument('--level', default='token')
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--cache_data', action='store_true')
    args = parser.parse_args()
    print(vars(args))

    print("Loading data...")
    prefix = '{level}.{min_len}.{min_freq}.{concat}'.format(**vars(args))
    if not args.cache_data or not os.path.isfile('data/%s_train.pt' % prefix):
        src, trg = load_twisty(
            min_len=args.min_len, level=args.level, concat=args.concat,
            processor=text_processor(lower=False))
        train, test, valid = load_dataset(
            src, trg, args.batch_size, min_freq=args.min_freq,
            gpu=args.gpu, dev=args.dev, test=args.test)
        if args.cache_data:
            train.to_disk('data/%s_train.pt' % prefix)
            test.to_disk('data/%s_test.pt' % prefix)
            valid.to_disk('data/%s_valid.pt' % prefix)
    else:
        train = PairedDataset.from_disk('data/%s_train.pt' % prefix)
        valid = PairedDataset.from_disk('data/%s_valid.pt' % prefix)
    train.set_gpu(args.gpu), valid.set_gpu(args.gpu)
    datasets = {'train': train, 'valid': valid}

    param_sampler = make_sampler({
        'emb_dim': ['uniform', int, 20, 50],
        'hid_dim': ['uniform', int, 20, 100],
        'dropout': ['loguniform', float, math.log(0.1), math.log(0.5)],
        'model': ['choice', str, ('CNNText', 'DCNN', 'RCNN')],
        'load_embeddings': ['choice', bool, (True, False)],
        'max_dim': ['uniform', int, 50, 200],
        # not applying to DCNN
        'out_channels': ['uniform', int, 10, 150],
        'kernel_sizes': ['choice', tuple, [
            (5, 4, 3), (7, 5, 4, 3), (9, 7, 5, 4, 3),
            (7, 5, 3), (9, 5, 3, 2), (12, 9, 6, 3)]],
        # only applying to DCNN: increase kernel_sizes, out_channels by factor
        'dcnn_factor': ['uniform', int, 1, 5], 'ktop': ['uniform', int, 3, 8],
        # 'lr': ['loguniform', float, math.log(0.001), math.log(0.05)]
    })

    vocab, n_classes = len(train.d['src'].vocab), len(train.d['trg'].vocab)

    def model_builder(params):
        if params['model'] == 'DCNN':
            kernel_sizes, out_channels = (7, 5), (6, 14),
            out_channels = tuple(
                [c * params['dcnn_factor'] for c in out_channels])
            kernel_sizes = tuple(
                [k * params['dcnn_factor'] for k in kernel_sizes])
        else:
            kernel_sizes = params['kernel_sizes']
            out_channels = params['out_channels']

        if params['load_embeddings']:
            weight = load_embeddings(
                train.d['src'].vocab, args.flavor, args.suffix, 'data')
            emb_dim = args.emb_dim
        else:
            emb_dim = params['emb_dim']

        model = getattr(models, args.model)(
            n_classes, vocab, emb_dim=emb_dim, hid_dim=params['hid_dim'],
            dropout=params['dropout'], padding_idx=train.d['src'].get_pad(),
            # cnn
            act=args.act, out_channels=out_channels, kernel_sizes=kernel_sizes,
            # - rcnn only
            max_dim=params['max_dim'],
            # - DCNN only
            ktop=params['ktop'])

        u.initialize_model(model)

        if params['load_embeddings']:
            if not args.load_embeddings:
                raise ValueError("Need load_embeddings")
            model.init_embeddings(weight)

        criterion = make_criterion(train)

        optimizer = Optimizer(
            model.parameters(), args.optim, lr=args.learning_rate,
            max_norm=args.max_norm, weight_decay=args.weight_decay)

        early_stopping = EarlyStopping(5, patience=3, reset_patience=False)

        def early_stop_hook(trainer, epoch, batch_num, num_checkpoints):
            valid_loss = trainer.merge_loss(trainer.validate_model())
            early_stopping.add_checkpoint(valid_loss)

        trainer = Trainer(model, datasets, criterion, optimizer)
        trainer.add_hook(make_score_hook(model, valid),
                         hooks_per_epoch=args.hooks_per_epoch)
        trainer.add_hook(early_stop_hook, hooks_per_epoch=5)

        def run(n_iters):
            batches = int(len(train) / args.max_iter) * 5
            if args.gpu:
                model.cuda(), criterion.cuda()
            (_, loss), _ = trainer.train_batches(
                batches, args.checkpoints, shuffle=True, gpu=args.gpu)
            model.cpu()
            return {'loss': loss, 'early_stop': early_stopping.stopped}

        return run

    hb = Hyperband(
        param_sampler, model_builder, max_iter=args.max_iter, eta=args.eta)
    print(hb.run())
