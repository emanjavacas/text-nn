
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTextNN(nn.Module):
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


class RCNN(BaseTextNN):
    """
    Implementation of Recurrent Convolutional Neural Network (RCNN)
    http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745

    The model runs a simple RNN on sentences adding a pooling layer
    across per-token (element-wise) activations in a convolutional
    fashion.

    Parameters:
    -----------
    - vocab: int, number of words in the vocabulary
    - emb_dim: int (e in the paper)
    - hid_dim: int, size of the hidden projections (c in the paper)
    - max_dim: int, size of the max pooled projection (H in the paper)
    - n_classes: int, number of classes in the classification
    """
    def __init__(self, n_classes, vocab, emb_dim=100, hid_dim=50, max_dim=100,
                 dropout=0.0, act='tanh', padding_idx=None, **kwargs):
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.cat_dim = (2 * hid_dim) + emb_dim  # c_l + emb + c_r
        self.max_dim = max_dim  # projection before max-pooling
        self.dropout = dropout
        self.act = getattr(F, act)
        super(RCNN, self).__init__()
        self.embeddings = nn.Embedding(
            self.vocab, self.emb_dim, padding_idx=padding_idx)
        self.c_l = nn.Parameter(torch.randn(self.hid_dim))
        self.c_r = nn.Parameter(torch.randn(self.hid_dim))
        self.W_l = nn.Linear(self.hid_dim, self.hid_dim)
        self.W_r = nn.Linear(self.hid_dim, self.hid_dim)
        self.W_sl = nn.Linear(self.emb_dim, self.hid_dim)
        self.W_sr = nn.Linear(self.emb_dim, self.hid_dim)
        self.max_proj = nn.Linear(self.cat_dim, self.max_dim)
        self.doc_proj = nn.Linear(self.max_dim, n_classes)

    def forward(self, inp):
        seq_len, batch = inp.size()

        # - run embeddings
        emb = self.embeddings(inp)

        # - build contexts (y)
        c_ls, c_rs = [], []
        emb_view = emb.view(seq_len * batch, self.emb_dim)
        emb_ls = self.W_sl(emb_view).view(seq_len, batch, self.hid_dim)
        emb_rs = self.W_sr(emb_view).view(seq_len, batch, self.hid_dim)
        c_l, c_r = self.c_l.repeat(batch, 1), self.c_r.repeat(batch, 1)
        for l_i in range(len(inp)):
            r_i = len(inp) - l_i - 1
            c_l = self.act(self.W_l(c_l) + emb_ls[l_i])
            c_r = self.act(self.W_r(c_r) + emb_rs[r_i])
            c_ls.append(c_l), c_rs.append(c_r)
        c_ls, c_rs = torch.stack(c_ls), torch.stack(c_rs[::-1])
        c_ls = F.dropout(c_ls, p=self.dropout, training=self.training)
        c_rs = F.dropout(c_rs, p=self.dropout, training=self.training)

        # - project concatenated contexts
        cat = torch.cat([c_ls, emb, c_rs], 2) \
                   .view(seq_len * batch, self.cat_dim)
        cat = F.dropout(cat, p=self.dropout, training=self.training)
        y = self.act(self.max_proj(cat)) \
                .view(seq_len, batch, self.max_dim)

        # - element-wise argmax over concatenated contexts
        y_max = torch.max(y, 0)[0].squeeze(0)  # (batch_size x cat_dim)
        return F.log_softmax(self.doc_proj(y_max))  # (batch_size x n_classes)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, act='relu'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.act = act
        super(ConvLayer, self).__init__()

        self.convs = []
        for d, (H, W) in enumerate(self.kernel_sizes):
            conv = nn.Conv2d(self.in_channels, self.out_channels, (H, W))
            self.add_module('ConvLayer_%d' % d, conv)
            self.convs.append(conv)

    def forward(self, inp):
        conv_out = []
        for conv in self.convs:
            # (batch x out_channels x 1 x W_out)
            out = conv(inp)
            # (batch x out_channels x W_out)
            out = out.squeeze(2)
            # (batch x out_channels x 1)
            out = F.max_pool1d(out, out.size(2))
            # (batch x out_channels)
            out = out.squeeze(2)
            out = getattr(F, self.act)(out)
            conv_out.append(out)
        conv_out = torch.stack(conv_out)  # (num_kernels x batch x out_channels)
        conv_out = conv_out.t() # (batch x num_kernels x out_channels)
        return conv_out


class CNNSent(BaseTextNN):
    """
    Implementation of Convolutional Neural Networks for Sentence Classification

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
                 out_channels=128, kernel_sizes=(5, 4, 3), act='relu',
                 # rest parameters
                 dropout=0.0, **kwargs):
        self.vocab = vocab
        self.emb_dim = emb_dim

        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.dropout = dropout

        super(CNNSent, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(
            self.vocab, self.emb_dim, padding_idx=padding_idx)

        # CNN
        self.conv = ConvLayer(
            in_channels=1,      # text has only one channel
            out_channels=self.out_channels,
            kernel_sizes=[(self.emb_dim, W) for W in self.kernel_sizes],
            act=act)

        # Projection
        proj_in = len(self.kernel_sizes) * self.out_channels
        self.proj = nn.Sequential(
            nn.Linear(proj_in, n_classes),
            nn.LogSoftmax())

    def forward(self, inp):
        # Embedding
        emb = self.embeddings(inp).t()  # (batch x seq_len x emb_dim)
        emb = emb.transpose(1, 2)       # (batch x emb_dim x seq_len)

        # CNN
        conv_out = self.conv(emb.unsqueeze(1))
        conv_out = F.dropout(
            conv_out, p=self.dropout, training=self.training)

        batch = conv_out.size(0)
        # (batch x num_kernels x out_channels)
        conv_out = conv_out.view(batch, -1)
        return self.proj(conv_out)


class ConvRec(BaseTextNN):
    """
    Efficient Char-level Document Classification
    by Combining Convolution and Recurrent Layers:
    https://arxiv.org/pdf/1602.00367.pdf

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
                 out_channels=128, kernel_sizes=(5, 3), pool_size=2, act='relu',
                 # rnn parameters
                 rnn_layers=1, hid_dim=128, cell='LSTM', bidi=True,
                 # rest parameters
                 dropout=0.0, **kwargs):
        self.vocab = vocab
        self.emb_dim = emb_dim

        self.out_channels = out_channels  # num_filters
        self.kernel_sizes = kernel_sizes  # filter_sizes
        self.pool_size = pool_size
        self.act = act

        self.hid_dim = hid_dim
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
        for d, W in enumerate(self.kernel_sizes):
            padding = math.floor(W / 2)
            H = self.emb_dim if d == 0 else self.out_channels
            conv = nn.Conv2d(1, self.out_channels, (H, W))
            self.add_module('Conv_%d' % d, conv)
            self.conv_layers.append(conv)
        
        # RNN
        self.rnn = getattr(nn, self.cell)(
            self.out_channels, self.hid_dim, self.rnn_layers,
            dropout=self.dropout, bidirectional=self.bidi)

        # Proj
        self.proj = nn.Sequential(
            nn.Linear(2 * self.hid_dim, n_classes),
            nn.LogSoftmax())

    def forward(self, inp):
        # Embedding
        emb = self.embeddings(inp).t()  # (batch x seq_len x emb_dim)
        emb = emb.transpose(1, 2)       # (batch x emb_dim x seq_len)
        emb = emb.unsqueeze(1)          # (batch x 1 x emb_dim x seq_len)

        # CNN
        conv_in = emb
        for conv_layer in self.conv_layers:
            # (batch x out_channels x 1 x seq_len)
            conv_out = conv_layer(conv_in)
            conv_out = getattr(F, self.act)(conv_out)
            # (batch x out_channels x 1 x floor(seq_len / 2))
            conv_out = F.max_pool2d(conv_out, (1, 2))
            conv_out = F.dropout(
                conv_out, p=self.dropout, training=self.training)
            # (batch x 1 x out_channels x floor(seq_len / 2))
            conv_in = conv_out.transpose(1, 2)

        # RNN
        # (reduced seq_len x batch x out_channels)
        rnn_in = conv_out.squeeze(2).t().transpose(0, 2).contiguous()
        # (seq_len x batch x hid_dim * 2)
        rnn_out, _ = self.rnn(rnn_in)

        # Proj
        # (batch x hid_dim * 2)
        rnn_out = rnn_out[-1, :, :]
        return self.proj(rnn_out)
