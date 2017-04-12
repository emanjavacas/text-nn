
import math
import argparse
import random
random.seed(1001)

import fasttext as ft
from sklearn.metrics import classification_report

from loaders import load_twisty


def permute(l):
    indices = list(range(len(l)))
    random.shuffle(indices)
    return indices


def write_lines(filename, src, trg):
        with open(filename, 'w') as f:
            for s, t in zip(src, trg):
                label = '{prefix}{label}'.format(
                    prefix=args['label_prefix'], label=t[0])
                f.write(label + ' , ' + ' '.join(s) + '\n')


def remove_keys(d, *keys):
    for k in keys:
        if k in d:
            del d[k]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--outputfile', default='ft_supervised')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lp_update_rate')
    parser.add_argument('--dim', default=100, type=int)
    parser.add_argument('--ws', default=5, type=int)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--min_count', default=1, type=int)
    parser.add_argument('--label_prefix', default='__label__')
    parser.add_argument('--neg', default=5, type=int)
    parser.add_argument('--word_ngrams', default=1, type=int)
    parser.add_argument('--loss', default='softmax')
    parser.add_argument('--bucket', default=0, type=int)
    parser.add_argument('--minn', default=0, type=int)
    parser.add_argument('--maxn', default=0, type=int)
    parser.add_argument('--thread', default=12, type=int)
    parser.add_argument('--t', default=0.0001, type=float)
    parser.add_argument('--load_model', action='store_true')
    # dataset
    parser.add_argument('--min_len', default=0, type=int)
    parser.add_argument('--min_freq', default=5, type=int)
    parser.add_argument('--level', default='token')
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--test', default=0.2, type=float)
    parser.add_argument('--load_data', action='store_true')

    args = vars(parser.parse_args())

    src, trg = load_twisty(
        min_len=args['min_len'], level='token', concat=args['concat'])
    perm = permute(src)
    split = math.floor(len(src) * args['test'])
    src_train, trg_train = src[split:], trg[split:]
    src_test, trg_test = src[:split], trg[:split]

    inputfile = '/tmp/fasttext_supervised_train'

    if args['load_model']:
        clf = ft.load_model(args['outputfile'] + '.bin', encoding='utf-8')
    else:
        write_lines(inputfile, src_train, trg_train)
        outputfile = args['outputfile']
        remove_keys(
            args, 'outputfile',
            'load_model', 'min_len', 'min_freq',
            'concat', 'test', 'load_data')
        clf = ft.supervised(inputfile, outputfile, args)

    y_true = [l[0] for l in trg_test]
    y_pred = clf.predict([' '.join(s) for s in src_test])
    print(classification_report(y_true, y_pred))
