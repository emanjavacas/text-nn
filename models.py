
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

    def init_embeddings(self, weight):
        if isinstance(weight, np.ndarray):
            self.embeddings.weight.data = torch.Tensor(weight)
        elif isinstance(weight, torch.Tensor):
            self.embeddings.weight.data = weight
        elif isinstance(weight, nn.Parameter):
            self.embeddings.weight = weight
        else:
            raise ValueError("Unknown weight type [%s]" % type(weight))

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


class ConvRec(nn.Module):
    """
    Implementation of Efficient Char-level Document Classification
    by Combining Convolution and Recurrent Layers:
    https://arxiv.org/pdf/1602.00367.pdf

    Parameters:
    -----------
    n_classes: int, number of classes
    vocab: int, vocabulary length
    emb_dim: int, embedding dimention
    out_channels: number of channels for all kernels
       This can't vary across filters their output congruency
    kernel_sizes: tuple of int, one for each kernel, i.e. number of kernels
       will equal the length of this argument. In practice, this parameter
       only controls the width of each filter since the height is fixed to
       the dimension of the embeddings to ensure a kernel output height of 1
       (the kernel output width will vary depending on the input seq_len,
       but will as well get max pooled over)
    hid_dim: int, RNN hidden dimension
    cell: str, one of 'LSTM', 'GRU', 'RNN'
    num_layers: int, depth of the RNN
    bidi: bool, whether to use bidirectional RNNs
    dropout: float
    act: str, activation function after the convolution
    padding_idx: int or None, zero the corresponding output embedding
    """
    def __init__(self, n_classes,
                 # embedding parameters
                 vocab, emb_dim=100,
                 # cnn parameters
                 out_channels=128, kernel_sizes=(5, 4, 3),
                 # rnn parameters
                 hid_dim=128, cell='LSTM', num_layers=1, bidi=True,
                 # rest parameters
                 dropout=0.0, act='ReLU', padding_idx=None, **kwargs):
        self.vocab = vocab
        self.emb_dim = emb_dim

        self.in_channels = 1    # text has only a channel
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.hid_dim = hid_dim
        self.cell = cell
        self.num_layers = num_layers
        self.bidi = bidi

        self.dropout = dropout
        self.act = act

        super(ConvRec, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(
            self.vocab, self.emb_dim, padding_idx=padding_idx)

        # CNN
        self.convs = [
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels,
                          # number of filters HxW for each kernel (H is
                          # fixed to emb_dim to ensure H_out equals one)
                          (self.emb_dim, kernel_size)),
                getattr(nn, self.act))
            for kernel_size in self.kernel_sizes]

        # RNN
        self.rnn = getattr(nn, self.cell)(
            self.out_channels, self.hid_dim, self.num_layers,
            dropout=self.dropout, bidirectional=self.bidi)

        # Projection
        self.proj = nn.Linear(
            2 * self.hid_dim,
            n_classes)

    def forward(self, inp):
        # Embedding
        emb = self.embeddings(inp).t()  # (batch x seq_len x emb_dim)
        emb = emb.transpose(1, 2).unsqueeze(1)  # (bs x 1 x emb_dim x seq_len)

        # CNN
        conv_out = []
        for conv in self.convs:
            # (batch x out_channels x 1 x W_out)
            # W_out = seq_len - kernel_size + 1
            out = conv(emb)
            # (batch x out_channels x 1)
            out = F.max_pool1d(out.squeeze(2), out.size(2))
            out = F.dropout(out, p=self.dropout, training=self.training)
            # (batch x out_channels)
            conv_out.append(out.squeeze(2))
        conv_out = torch.stack(conv_out)  # (num_kernels x bs x out_channels)
        conv_out = F.dropout(
            conv_out, dropout=self.dropout, training=self.training)

        # RNN
        rnn_out, _ = self.rnn(conv_out)
        # take only last layer & last state output
        return F.log_softmax(self.proj(rnn_out[-1]))
