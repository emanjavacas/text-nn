
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from text_nn.models.base import BaseTextNN


class CNNText(BaseTextNN):
    """
    'Convolutional Neural Networks for Sentence Classification'
    http://www.aclweb.org/anthology/D14-1181

    Parameters:
    -----------
    n_classes: int, number of classes
    vocab: int, vocabulary length
    emb_dim: int, embedding dimension
    padding_idx: int or None, zero the corresponding output embedding
    out_channels: number of channels for all kernels
       This can't vary across filters their output congruency
    kernel_sizes: tuple of int, one for each kernel, i.e. number of kernels
       will equal the length of this argument. In practice, this parameter
       only controls the width of each filter since the height is fixed to
       the dimension of the embeddings to ensure a kernel output height of 1
       (the kernel output width will vary depending on the input seq_len,
       but will as well get max pooled over)
    act: str, activation function after the convolution
    dropout: float
    """
    def __init__(self, n_classes,
                 # embedding parameters
                 vocab, emb_dim=100, padding_idx=None,
                 # cnn parameters
                 out_channels=100, kernel_sizes=(5, 4, 3), act='relu',
                 conv_type='wide',
                 # rest parameters
                 dropout=0.0, **kwargs):
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.act = act
        self.dropout = dropout
        self.conv_type = conv_type

        super(CNNText, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(
            self.vocab, self.emb_dim, padding_idx=padding_idx)

        # CNN
        padding, conv = 0, []
        W, C_i, C_o = emb_dim, 1, out_channels
        for H in kernel_sizes:
            if self.conv_type == 'wide':
                padding = math.floor(H / 2) * 2
            conv.append(nn.Conv2d(C_i, C_o, (H, W), padding=(padding, 0)))
        self.conv = nn.ModuleList(conv)

        # Projection
        proj_in = len(kernel_sizes) * out_channels
        self.proj = nn.Sequential(
            nn.Linear(proj_in, n_classes),
            nn.LogSoftmax())

    def _maybe_add_padding(self, inp):
        if self.conv_type == 'wide':
            return inp

        seq_len, batch = inp.size()

        # input sequence needs padding
        if seq_len - max(self.kernel_sizes) < 0:
            # check if embeddings have entry for padding
            padding_idx = self.embeddings.padding_idx
            if padding_idx is None:
                raise ValueError("Needs padding for too small input")
            # concat padding to the input (left-side)
            pad_size = max(self.kernel_sizes) - seq_len, batch
            pad = Variable(inp.data.new(*pad_size).zero_()) + padding_idx
            inp = torch.cat([pad, inp], dim=0)

        return inp

    def forward(self, inp):
        inp = self._maybe_add_padding(inp)

        # Embedding
        emb = self.embeddings(inp)
        emb = emb.transpose(0, 1)  # (batch x seq_len x emb_dim)
        emb = emb.unsqueeze(1)     # (batch x 1 x seq_len x emb_dim)

        # CNN
        conv_outs = []
        for conv in self.conv:
            conv_out = conv(emb)
            # (batch x Ci x seq_len x 1)
            conv_out = getattr(F, self.act)(conv_out).squeeze(3)
            # (batch x Ci)
            conv_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outs.append(conv_out)

        conv_outs = torch.cat(conv_outs, dim=1)

        conv_outs = F.dropout(
            conv_outs, p=self.dropout, training=self.training)

        # (batch x num_kernels * out_channels)
        return self.proj(conv_outs)
