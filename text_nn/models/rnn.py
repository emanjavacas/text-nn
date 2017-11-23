
import torch.nn as nn

from models.base import BaseTextNN


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
        self.hid_dim = hid_dim
        self.bidi
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
            bidirectional=self.bidi, dropout=self.dropout)

        # proj
        self.proj = nn.Sequential(
            nn.Linear(self.hid_dim * self.dirs, n_classes),
            nn.LogSoftmax())

    def init_hidden_for(self, inp):
        batch = inp.size(1)
        size = (self.dirs * self.num_layers, batch, self.hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        if self.cell.startswith('LSTM'):
            c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
            return h_0, c_0
        else:
            return h_0

    def forward(self, inp):
        # Embedding
        emb = self.embeddings(inp).t()  # (batch x seq_len x emb_dim)
        outs, hidden = self.rnn(inp, hidden or self.init_hidden_for(inp))
        outs = outs.view(self.num_layers, self.dirs, -1, self.hid_dim)
        # (batch x dirs * hid_dim)
        outs = outs[-1].t().view(-1, self.dirs * self.hid_dim)
        return self.proj(outs)
