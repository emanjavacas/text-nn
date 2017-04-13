
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseTextNN


def dynamic_ktop(l, L, s, min_ktop):
    """
    Computes the ktop parameter for layer l in a stack of L layers
    across a sequence of length s.
    """
    return max(min_ktop, math.ceil(((L - l) / L) * s))


def global_kmax_pool(t, ktop, dim=3):
    """
    Return the ktop max elements for each feature across dimension 3
    respecting the original order.
    """
    _, indices = t.topk(ktop, dim=dim)
    indices, _ = indices.sort(dim=dim)
    return t.gather(3, indices)


def folding(t, factor=2):
    """
    Applies folding across the height (embedding features) of the feature
    maps of a given convolution. Folding can be seen as applying local
    sum-pooling across the feature dimension (instead of the seq_len dim).
    """
    rows = [fold.sum(2) for fold in t.split(factor, dim=2)]
    return torch.cat(rows, 2)


class DCNN(BaseTextNN):
    """
    Implementation of 'A Convolutional NN for Modelling Sentences'
    https://arxiv.org/pdf/1404.2188.pdf

    Multi-layer CNN with Dynamic K-max pooling and folding.

    Convolutions between 1d-filters (d x m) and the embedding sentence
    matrix (d x s) are applied in 2-d yielding a matrix (d x (s + m - 1))
    or (d x (s - m + 1) depending on whether wide or narrow 
    convolutions are used (the difference being in using padding or not).
    After each convolutional layer, the top k features of the resulting
    feature map are taken row-wise (e.g. the number of top k operations
    is equal to the embedding dimension d) resulting in a subsampling down
    to k. k is dynamically computed by a non-learned function (see ktop).
    """
    def __init__(self, n_classes,
                 # embeddings
                 vocab, emb_dim=24, padding_idx=None,
                 # cnn
                 kernel_sizes=(7, 5), out_channels=(6, 14),
                 ktop=4, folding=2, conv_type='wide', act='tanh',
                 # rest parameters
                 dropout=0.0, **kwargs):
        if len(kernel_sizes) != len(out_channels):
            warnings.warn("""
            The number of layers according to kernel_sizes doesn't match
            those according to out_channels. kernel_sizes will be truncated
            to the length of out_channels.""")

        self.vocab = vocab
        self.emb_dim = emb_dim

        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.ktop = ktop
        self.folding = folding
        self.conv_type = conv_type
        self.act = act

        self.dropout = dropout

        super(DCNN, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(
            self.vocab, self.emb_dim, padding_idx=padding_idx)

        # CNN
        H = self.emb_dim      # variable to get H after all CNN layers
        self.conv_layers = []
        for d, (C_o, K) in enumerate(zip(self.out_channels, self.kernel_sizes)):
            # text only has one channel
            C_i = 1 if d == 0 else self.out_channels[d - 1]
            # feature map H gets folded each layer
            H = math.ceil(H / self.folding)
            # wide(filter_size) = full(filter_size) * 2
            pad = math.floor(K / 2) * 2 if self.conv_type == 'wide' else 0
            conv = nn.Conv2d(C_i, C_o, (1, K), padding=(0, pad))
            self.add_module('Conv2d_%d' % d, conv)
            self.conv_layers.append(conv)
        H = math.ceil(H / folding)  # last folding after convolutions

        # Projection
        proj_in = self.out_channels[-1] * H * self.ktop
        self.proj = nn.Sequential(
            nn.Linear(proj_in, n_classes),
            nn.LogSoftmax())

    def forward(self, inp):
        # Embedding
        emb = self.embeddings(inp).t()  # (batch x seq_len x emb_dim)
        emb = emb.transpose(1, 2)       # (batch x emb_dim x seq_len)
        emb = emb.unsqueeze(1)          # (batch x 1 x emb_dim x seq_len)

        # CNN
        conv_in = emb
        for l, conv_layer in enumerate(self.conv_layers):
            # (batch x out_channels x (emb_dim\prev-kernel_size) x (seq_len + pad))
            conv_out = conv_layer(conv_in)
            # (batch x out_channels x (emb_dim\prev-kernel_sizes/2) x (seq_len + pad))
            conv_out = folding(conv_out)
            L, s = len(self.conv_layers), conv_out.size(3)
            # (batch x out_channels x (emb_dim\prev-kernel_sizes/2) x ktop)
            conv_out = global_kmax_pool(conv_out, dynamic_ktop(l, L, s, self.ktop))
            conv_out = getattr(F, self.act)(conv_out)
            conv_in = conv_out

        conv_out = F.dropout(
            conv_out, p=self.dropout, training=self.training)

        # Final k-max
        conv_out = global_kmax_pool(conv_out, self.ktop)
        conv_out = folding(conv_out)

        # Projection
        batch = conv_out.size(0)
        proj_in = conv_out.view(batch, -1)
        return self.proj(proj_in)
