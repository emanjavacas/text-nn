
from collections import defaultdict
import json

import visdom
import numpy as np


def load_db(path="db.json"):
    with open(path, 'r') as f:
        return json.load(f)


def get_experiment_values(exp_id, metric, **kwargs):
    db = load_db(**kwargs)
    out = defaultdict(list)
    for exp in db['_default'].values():
        if exp['id'] == exp_id:
           for model in exp['models']:
               model_id = model['modelId']
               for sess in model['sessions']:
                   min_len = sess['params']['min_len']
                   score = sess['result']['class_' + metric]
                   out[model_id].append((float(min_len), float(score)))
    legend, X, y = [], [], []
    for k, items in out.items():
        items = sorted(items, key=lambda x: x[0])
        if len(y) == 0:
            y = [y for y, _ in items]
        if len(legend) < len(items):
            legend.append(k)
        if not X:
            X = [[x] for _, x in items]
        else:
            for idx, (_, x) in enumerate(items):
                X[idx].append(x)
    return legend, X, y


def plot_experiment(
        exp_id, metric,
        server='http://146.175.11.197', port=8097, env='gender', **kwargs):
    legend, X, y = get_experiment_values(exp_id, metric, **kwargs)
    vis = visdom.Visdom(server=server, port=port, env=env)
    vis.line(np.array(X), np.array(y),
             opts={'legend': legend, 'title': exp_id + ':' + metric})
