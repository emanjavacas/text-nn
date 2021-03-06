
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from text_nn.models.base import BaseTextNN


class ConvRec(BaseTextNN):
    """
    Efficient Char-level Document Classification
    by Combining Convolution and Recurrent Layers:
    https://arxiv.org/pdf/1602.00367.pdf

    WARNING:
    From the paper it is still unclear what the feature dimension after
    the convolutions is defined. I am assuming that this corresponds to
    the number of feature maps (out_channels) after the last layer - which
    is, anyway, held constant across conv layers.

    It also deviates from the original description in that we use the
    concatenation of the last step and the average n-1 first steps of
    the RNN as input to the softmax projection layer. This has the goal
    of improving gradient flow through the RNN layer.

    Parameters:
    -----------
    n_classes: int, number of classes
    vocab: int, vocabulary length
    emb_dim: int, embedding dimension
    padding_idx: int or None, zero the corresponding output embedding
    out_channels: int, number of filters per layers (out_channels?)
    kernel_sizes: tuple or list of int, filter size for the filters per layer
        The length of this parameters implicitely determines the number of
        convolutional layers.
    pool_size: int, size of horizontal max pooling
    hid_dim: int, RNN hidden dimension
    cell: str, one of 'LSTM', 'GRU', 'RNN'
    bidi: bool, whether to use bidirectional RNNs
    dropout: float
    act: str, activation function after the convolution
    """
    def __init__(self, n_classes,
                 # embeddings
                 vocab, emb_dim=100, padding_idx=None,
                 # cnn
                 out_channels=128, kernel_sizes=(5, 3), pool_size=2,
                 act='relu', conv_type='wide',
                 # rnn parameters
                 rnn_layers=1, hid_dim=128, cell='LSTM', bidi=True,
                 # rest parameters
                 dropout=0.0, **kwargs):
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.out_channels = out_channels  # num_filters
        self.kernel_sizes = kernel_sizes  # filter_sizes
        self.pool_size = pool_size
        self.conv_type = conv_type
        self.act = act

        self.hid_dim = hid_dim // 2 if bidi else hid_dim
        self.cell = cell
        self.bidi = bidi
        self.rnn_layers = rnn_layers

        self.dropout = dropout

        super(ConvRec, self).__init__()
        
        # Embeddings
        self.embeddings = nn.Embedding(
            self.vocab, self.emb_dim, padding_idx=padding_idx)

        # CNN
        self.conv_layers = []
        for layer, W in enumerate(self.kernel_sizes):
            C_i, H = None, None
            if layer == 0:
                C_i, H = 1, self.emb_dim
            else:
                C_i, H = self.out_channels, 1
            padding = math.floor(W / 2) * 2 if self.conv_type == 'wide' else 0

            conv = nn.Conv2d(
                C_i, self.out_channels, (H, W), padding=(0, padding))
            self.add_module('Conv_{}'.format(layer), conv)
            self.conv_layers.append(conv)

        # RNN
        self.rnn = getattr(nn, self.cell)(
            self.out_channels, self.hid_dim, self.rnn_layers,
            dropout=self.dropout, bidirectional=self.bidi)

        # Proj
        self.proj = nn.Sequential(
            nn.Linear(2 * self.hid_dim * (1 + int(bidi)), n_classes),
            nn.LogSoftmax())

    def forward(self, inp):        
        # Embedding
        emb = self.embeddings(inp)
        emb = emb.transpose(0, 1)  # (batch x seq_len x emb_dim)
        emb = emb.transpose(1, 2)  # (batch x emb_dim x seq_len)
        emb = emb.unsqueeze(1)     # (batch x 1 x emb_dim x seq_len)

        # CNN
        conv_in = emb
        for conv_layer in self.conv_layers:
            # (batch x out_channels x 1 x seq_len)
            conv_out = conv_layer(conv_in)
            # (batch x out_channels x 1 x floor(seq_len / pool_size))
            conv_out = F.max_pool2d(conv_out, (1, self.pool_size))
            conv_out = getattr(F, self.act)(conv_out)
            conv_out = F.dropout(
                conv_out, p=self.dropout, training=self.training)
            conv_in = conv_out

        # RNN
        # (floor(seq_len / pool_size) x batch x out_channels)
        rnn_in = conv_out.squeeze(2) \
                         .transpose(0, 1) \
                         .transpose(0, 2).contiguous()
        # (floor(seq_len / pool_size) x batch x hid_dim * 2)
        rnn_out, _ = self.rnn(rnn_in)
        # - average n-1 steps for better gradient flow (batch x hid_dim * 2)
        rnn_out = torch.cat([rnn_out[:-1].sum(0), rnn_out[-1]], dim=1)

        # Proj
        out = self.proj(rnn_out)

        return out
