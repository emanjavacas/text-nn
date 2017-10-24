
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTextNN(nn.Module):
    def __init__(self, *args, weight=None, **kwargs):
        self.weight = None
        if weight is not None:
            self.weight = torch.Tensor(weight)

    def init_embeddings(self, weight):
        if isinstance(weight, np.ndarray):
            self.embeddings.weight.data = torch.Tensor(weight)
        elif isinstance(weight, torch.Tensor):
            self.embeddings.weight.data = weight
        elif isinstance(weight, nn.Parameter):
            self.embeddings.weight = weight
        else:
            raise ValueError("Unknown weight type [%s]" % type(weight))

    def predict(self, inp):
        log_probs = self(inp)
        scores, preds = log_probs.max(1)
        return scores, preds

    def loss(self, batch, test=True):
        src, trg = batch

        output = self(src)
        loss = F.nll_loss(output, trg, weight=self.weight, size_average=True)

        if test:
            loss.backward()

        return (loss.data[0], ), len(trg)
