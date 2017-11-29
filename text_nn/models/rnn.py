
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from text_nn.models.base import BaseTextNN


class RNNText(BaseTextNN):
    def __init__(self, n_classes,
                 # embedding parameters
                 vocab, emb_dim=100, padding_idx=None,
                 # rnn parameters
                 num_layers=1, hid_dim=100, cell='LSTM', bidi=True,
                 # rest parameters
                 dropout=0.0, **kwargs):
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.hid_dim = hid_dim // 2 if bidi else hid_dim
        self.dirs = 2 if bidi else 1
        self.cell = cell

        self.dropout = dropout

        super(RNNText, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(
            self.vocab, self.emb_dim, padding_idx=padding_idx)

        # RNN
        self.rnn = getattr(nn, cell)(
            self.emb_dim, self.hid_dim, num_layers=self.num_layers,
            bidirectional=bidi, dropout=self.dropout)

        # proj
        self.proj = nn.Sequential(
            nn.Linear(self.hid_dim * self.dirs * 2, n_classes),
            nn.LogSoftmax())

    def init_hidden_for(self, inp):
        batch = inp.size(1)
        size = (self.dirs * self.num_layers, batch, self.hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        if self.cell.startswith('LSTM'):
            c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
            return h_0.float(), c_0.float()
        else:
            return h_0.float()

    def forward(self, inp, hidden=None):
        # Embedding
        emb = self.embeddings(inp) # (seq_len x batch x emb_dim)
        # (seq_len x batch x hid_dim)
        context, _ = self.rnn(emb, hidden or self.init_hidden_for(inp))

        context = F.dropout(context, p=self.dropout, training=self.training)
        # - concat last step and the average of the previous n-1 steps
        # (batch x hid_dim * dirs * 2)
        context = torch.cat([context[-1], context[:-1].mean(0)], 1)

        return self.proj(context)
