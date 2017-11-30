
import os
from collections import Counter
import argparse

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

from seqmod.misc.preprocess import text_processor
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.trainer import Trainer
from seqmod.misc.loggers import StdLogger
from seqmod.misc.dataset import PairedDataset
from seqmod.misc.early_stopping import EarlyStopping
import seqmod.utils as u

from text_nn.loaders import load_twisty, load_dataset, load_embeddings

from text_nn import models


def compute_scores(model, dataset):
    trues, preds = [], []
    for src, trg in dataset:
        _, pred = model.predict(src)
        trues.extend(trg.view(-1).data.tolist())
        preds.extend(pred.view(-1).data.tolist())
    return trues, preds


def make_score_hook(model, dataset):

    def hook(trainer, epoch, batch, checkpoint):
        trues, preds = compute_scores(model, dataset)
        trainer.log("info", classification_report(trues, preds))

    return hook


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model', required=True)
    parser.add_argument('--emb_dim', default=50, type=int)
    parser.add_argument('--hid_dim', default=50, type=int)
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--load_embeddings', action='store_true')
    parser.add_argument('--flavor', default=None)
    parser.add_argument('--suffix', default=None)
    parser.add_argument('--max_dim', default=100, type=int)
    parser.add_argument('--out_channels', default=(12,), nargs='+', type=int)
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=(5, 4, 3))
    parser.add_argument('--act', default='relu')
    parser.add_argument('--ktop', default=4, type=int)
    # training
    parser.add_argument('--optim', default='Adagrad')
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=264)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--outputfile', default=None)
    parser.add_argument('--checkpoints', default=100, type=int)
    parser.add_argument('--exp_id', default='')
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
            min_len=args.min_len, concat=args.concat,
            processor=text_processor(num=False, level=args.level),
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
        test = PairedDataset.from_disk('data/{}_test.pt'.format(prefix))
        valid = PairedDataset.from_disk('data/{}_valid.pt'.format(prefix))
        train.set_gpu(args.gpu)
        test.set_gpu(args.gpu)
        valid.set_gpu(args.gpu)

    print('* number of train batches. {}'.format(len(train)))
    datasets = {'train': train, 'valid': valid, 'test': test}

    class_weight = Counter(train.data['trg'])
    class_weight = [v for k, v in sorted(class_weight.items())]

    print("Building model...")
    kernel_sizes, out_channels = args.kernel_sizes, args.out_channels
    if args.model != 'DCNN':
        out_channels = out_channels[0]
    # else:
    #     kernel_sizes, out_channels = (7, 5), (6, 14)

    model = getattr(models, args.model)(
        len(train.d['trg'].vocab), len(train.d['src'].vocab),
        weight=class_weight,    # weight loss by class
        emb_dim=args.emb_dim, hid_dim=args.hid_dim,
        dropout=args.dropout, padding_idx=train.d['src'].get_pad(),
        # cnn
        act=args.act, out_channels=out_channels, kernel_sizes=kernel_sizes,
        # - rcnn only
        max_dim=args.max_dim,
        # - DCNN only
        ktop=args.ktop)

    u.initialize_model(model)

    print(model)
    n_params = sum(w.nelement() for w in model.parameters())
    print('* number of parameters. {}'.format(n_params))

    if args.load_embeddings:
        emb_weight = load_embeddings(
            train.d['src'].vocab, args.flavor, args.suffix, 'data')
        model.init_embeddings(emb_weight)

    if args.gpu:
        model.cuda()

    optimizer = Optimizer(
        model.parameters(), args.optim, lr=args.learning_rate,
        max_norm=args.max_norm, weight_decay=args.weight_decay)

    early_stopping = None
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(10, patience=args.early_stopping)

    trainer = Trainer(model, datasets, optimizer, early_stopping=early_stopping)
    trainer.add_loggers(StdLogger(args.outputfile))
    trainer.add_hook(make_score_hook(model, valid), hooks_per_epoch=1)
    trainer.train(args.epochs, args.checkpoints, shuffle=True)

    test_true, test_pred = compute_scores(model, test)
    trainer.log("info", classification_report(test_true, test_pred))

    if args.exp_id:
        from casket import Experiment
        db = Experiment.use('db.json', exp_id=args.exp_id).model(args.model)
        p, r, f, s = precision_recall_fscore_support(test_true, test_pred)
        db.add_result({'precision': p.tolist(), 'recall': r.tolist(),
                       'fscore': f.tolist(), 'support': s.tolist(),
                       'class_precision': np.average(p, weights=s),
                       'class_recall': np.average(r, weights=s),
                       'class_fscore': np.average(f, weights=s),
                       'n_params': n_params},
                      params=vars(args))

