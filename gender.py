
import os

from collections import Counter
import argparse

from sklearn import metrics
import torch
import torch.nn as nn

from misc.preprocess import text_processor
from misc.optimizer import Optimizer
from misc.trainer import Trainer
from misc.loggers import StdLogger, VisdomLogger
from misc.dataset import PairedDataset
from modules import utils as u

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
    parser.add_argument('--model', required=True)
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
    parser.add_argument('--ktop', default=6, type=int)
    # training
    parser.add_argument('--optim', default='Adagrad')
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--outputfile', default=None)
    parser.add_argument('--checkpoints', default=100, type=int)
    parser.add_argument('--exp_id', default='test')
    # dataset
    parser.add_argument('--dev', default=0.1, type=float)
    parser.add_argument('--test', default=0.2, type=float)
    parser.add_argument('--min_len', default=0, type=int)
    parser.add_argument('--min_freq', default=5, type=int)
    parser.add_argument('--level', default='token')
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--cache_data', action='store_true')
    args = parser.parse_args()

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
        test = PairedDataset.from_disk('data/%s_test.pt' % prefix)
        valid = PairedDataset.from_disk('data/%s_valid.pt' % prefix)
        train.set_gpu(args.gpu), test.set_gpu(args.gpu), valid.set_gpu(args.gpu)
    datasets = {'train': train, 'valid': valid, 'test': test}

    print("Building model...")
    vocab, n_classes = len(train.d['src'].vocab), len(train.d['trg'].vocab)
    if args.model != 'DCNN':
        out_channels = args.out_channels[0]
    else:
        out_channels = args.out_channels

    model = getattr(models, args.model)(
        n_classes, vocab,
        emb_dim=args.emb_dim, hid_dim=args.hid_dim,
        dropout=args.dropout, padding_idx=train.d['src'].get_pad(),
        # cnn
        out_channels=out_channels, kernel_sizes=args.kernel_sizes,
        # - rcnn only
        max_dim=args.max_dim,
        # - DCNN only
        ktop=args.ktop)
    model.apply(u.make_initializer())
    print(model)

    if args.load_embeddings:
        weight = load_embeddings(
            train.d['src'].vocab, args.flavor, args.suffix, 'data')
        model.init_embeddings(weight)

    criterion = make_criterion(train)
    if args.gpu:
        model.cuda(), criterion.cuda()

    optimizer = Optimizer(
        model.parameters(), args.optim, lr=args.learning_rate,
        max_norm=args.max_norm, weight_decay=args.weight_decay)

    trainer = Trainer(model, datasets, criterion, optimizer)
    trainer.log('info', '* number of parameters. %d' %
                len(list(model.parameters())))
    trainer.log('info', '* number of train batches/examples. %d/%d' %
                (len(train), len(train) * args.batch_size))
    std = StdLogger(args.outputfile)
    title = '{model}.{min_len}.{level}'.format(**vars(args))
    visdom = VisdomLogger(env='gender', title=title)
    trainer.add_loggers(std, visdom)
    checks = max(len(train) // args.checkpoints, 1)
    trainer.add_hook(make_score_hook(model, valid), num_checkpoints=checks)
    trainer.train(args.epochs, args.checkpoints, shuffle=True, gpu=args.gpu)

    test_true, test_pred = compute_scores(model, test)
    trainer.log("info", metrics.classification_report(test_true, test_pred))

    from casket import Experiment
    db = Experiment.use('db.json', exp_id=args.exp_id).model('RCNN')
    db.add_result({'acc': metrics.accuracy_score(test_true, test_pred),
                   'auc': metrics.roc_auc_score(test_true, test_pred),
                   'f1': metrics.f1_score(test_true, test_pred),
                   'test_examples': len(test_true)},
                  params=vars(args))
