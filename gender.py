
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

from loaders import load_twisty, load_dataset, load_embeddings
from rcnn import RCNN


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
        print(metrics.classification_report(trues, preds))

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
    parser.add_argument('--max_dim', default=100, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--load_embeddings', action='store_true')
    parser.add_argument('--flavor', default=None)
    parser.add_argument('--suffix', default=None)
    # training
    parser.add_argument('--optim', default='Adam')
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--outputfile', default=None)
    parser.add_argument('--checkpoints', default=100, type=int)
    # dataset
    parser.add_argument('--dev', default=0.1, type=float)
    parser.add_argument('--test', default=0.2, type=float)
    parser.add_argument('--min_len', default=0, type=int)
    parser.add_argument('--min_freq', default=5, type=int)
    parser.add_argument('--level', default='token')
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--cache_data', action='store_true')
    parser.add_argument('--load_data', action='store_true')
    args = parser.parse_args()

    print("Loading data...")
    if args.load_data:
        train = PairedDataset.from_disk('data/train.pt')
        test = PairedDataset.from_disk('data/test.pt')
        valid = PairedDataset.from_disk('data/valid.pt')
    else:
        src, trg = load_twisty(
            min_len=args.min_len, level=args.level, concat=args.concat,
            processor=text_processor(lower=False))
        train, test, valid = load_dataset(
            src, trg, args.batch_size, min_freq=args.min_freq,
            gpu=args.gpu, dev=args.dev, test=args.test)
    if args.cache_data:
        assert not args.load_data, "Data is already cached"
        train.to_disk('data/train.pt')
        test.to_disk('data/test.pt')
        valid.to_disk('data/valid.pt')
    datasets = {'train': train, 'valid': valid, 'test': test}

    print("Building model...")
    vocab, n_classes = len(train.d['src'].vocab), len(train.d['trg'].vocab)
    model = RCNN(vocab, args.emb_dim, args.hid_dim, args.max_dim, n_classes,
                 dropout=args.dropout)
    if args.load_embeddings:
        weight = load_embeddings(
            train.d['src'].vocab, args.flavor, args.suffix, 'data')
        model.init_embeddings(weight)
    print(model)

    criterion = make_criterion(train)
    if args.gpu:
        model.cuda(), criterion.cuda()

    optimizer = Optimizer(model.parameters(), args.optim,
                          lr=args.learning_rate, max_norm=args.max_norm)

    print("Training...")
    trainer = Trainer(model, datasets, criterion, optimizer)
    trainer.log('info', '* number of parameters. %d' %
                len(list(model.parameters())))
    trainer.log('info', '* number of train batches/examples. %d/%d' %
                (len(train), len(train) * args.batch_size))
    std, visdom = StdLogger(args.outputfile), VisdomLogger(env='gender')
    trainer.add_loggers(std, visdom)
    checks = max(len(train) // args.checkpoints, 1)
    trainer.add_hook(make_score_hook(model, valid), num_checkpoints=checks)
    trainer.train(args.epochs, args.checkpoints, shuffle=True, gpu=args.gpu)

    print("Testing...")
    test_true, test_pred = compute_scores(model, test)
    print(metrics.classification_report(test_true, test_pred))
