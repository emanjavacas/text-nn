
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from text_nn.models.base import BaseTextNN


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
    return torch.stack(rows, 2)


def get_padding(filter_size, mode='wide'):
    """
    Get padding for the current convolutional layer according to different
    schemes.

    Parameters:
    -----------
    filter_size: int
    mode: str, one of 'wide', 'narrow'
    """
    pad = 0
    if mode == 'wide':
        pad = math.floor(filter_size / 2) * 2

    return pad


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
                 ktop=4, folding_factor=2, conv_type='wide', act='tanh',
                 # rest parameters
                 dropout=0.0, **kwargs):
        if len(kernel_sizes) != len(out_channels):
            raise ValueError("Need same number of feature maps for "
                             "`kernel_sizes` and `out_channels`")

        self.vocab = vocab
        self.emb_dim = emb_dim

        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.ktop = ktop
        self.folding_factor = folding_factor
        self.conv_type = conv_type
        self.act = act
        self.dropout = dropout

        super(DCNN, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(
            self.vocab, self.emb_dim, padding_idx=padding_idx)

        # CNN
        # variable to get H after all CNN layers
        H, self.conv_layers = self.emb_dim, []

        for l, (C_o, K) in enumerate(zip(self.out_channels, self.kernel_sizes)):
            # text only has one channel
            C_i = 1 if l == 0 else self.out_channels[l-1]
            # feature map H gets folded each layer
            H = math.ceil(H / self.folding_factor)
            # 1D convolutions with multiple filters
            conv = nn.Conv2d(
                C_i, C_o, (1, K), padding=(0, get_padding(K, self.conv_type)))
            # add layer
            self.add_module('Conv1d_{}'.format(l), conv)
            self.conv_layers.append(conv)

        # Projection
        proj_in = self.out_channels[-1] * H * self.ktop
        self.proj = nn.Sequential(
            nn.Linear(proj_in, n_classes),
            nn.LogSoftmax())

    def forward(self, inp):
        # Embedding
        emb = self.embeddings(inp)
        emb = emb.transpose(0, 1)  # (batch x seq_len x emb_dim)
        emb = emb.transpose(1, 2)  # (batch x emb_dim x seq_len)
        emb = emb.unsqueeze(1)     # (batch x 1 x emb_dim x seq_len)

        # CNN
        conv_in = emb
        for l, conv_layer in enumerate(self.conv_layers):
            # - num_kernels: number of kernels run in parallel
            # - feat_dim: number of features (gets folded over every layer)
            # - s: output length of the convs (input seq_len + padding_idx)

            # (batch x num_kernels x feat_dim x (seq_len + pad))
            conv_out = conv_layer(conv_in)
            # (batch x num_kernels x (feat_dim / 2) x (seq_len + pad))
            conv_out = folding(conv_out)
            # - dynamic k-max
            L, s = len(self.conv_layers), conv_out.size(3)  # s: current length
            ktop = dynamic_ktop(l+1, L, s, self.ktop)
            # (batch x num_kernels x (feat_dim / 2) x ktop)
            conv_out = global_kmax_pool(conv_out, ktop)
            conv_out = getattr(F, self.act)(conv_out)
            conv_in = conv_out

        # Apply dropout to the penultimate layer after the last non-linearity
        conv_out = F.dropout(
            conv_out, p=self.dropout, training=self.training)

        # Projection (+ softmax)
        proj_in = conv_out.view(conv_out.size(0), -1)  # (batch x proj input)
        proj_out = self.proj(proj_in)                  # (batch x n_classes)

        return proj_out
