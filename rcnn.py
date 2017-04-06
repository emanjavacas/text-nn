
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RCNN(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, max_dim, n_classes,
                 num_layers=1, cell='LSTM', act='tanh', padding_idx=None):
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.cell = cell
        self.num_layers = num_layers
        self.cat_dim = (2 * hid_dim) + emb_dim  # c_l + emb + c_r
        self.max_dim = max_dim  # projection before max-pooling
        self.act = getattr(F, act)
        super(RCNN, self).__init__()
        self.embeddings = nn.Embedding(
            self.vocab, self.emb_dim, padding_idx=padding_idx)
        self.rnn = getattr(nn, cell)(
            self.hid_dim, num_layers=self.num_layers, bidirectional=True)
        self.c_l = nn.Parameter(torch.randn(self.hid_dim))
        self.c_r = nn.Parameter(torch.randn(self.hid_dim))
        self.W_l = nn.Linear(self.hid_dim, self.hid_dim)
        self.W_r = nn.Linear(self.hid_dim, self.hid_dim)
        self.W_sl = nn.Linear(self.emb_dim, self.hid_dim)
        self.W_sr = nn.Linear(self.hid_dim, self.emb_dim)
        self.max_proj = nn.Linear(self.cat_dim, self.max_dim)
        self.doc_proj = nn.Linear(self.max_dim, n_classes)

    def init_hidden_for(self, inp):
        batch = inp.size(1)
        size = (2 * self.num_layers, batch, self.hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        if self.cell.startswith('LSTM'):
            c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
            return h_0, c_0
        else:
            return h_0

    def forward(self, inp):
        seq_len, batch = inp.size()
        emb = self.embeddings(inp)
        outs, hidden = self.rnn(emb, self.init_hidden_for(inp)) \
                           .view(seq_len, batch, self.hid_dim, 2)

        # - build recursive embeddings (y)
        c_ls = Variable(inp.data.new(seq_len, batch, self.hid_dim))
        c_rs = Variable(inp.data.new(seq_len, batch, self.hid_dim))
        emb_view = emb.view(seq_len * batch, self.emb_dim)
        emb_ls = self.W_sl(emb_view).view(seq_len, batch, self.hid_dim)
        emb_rs = self.W_sr(emb_view).view(seq_len, batch, self.hid_dim)
        c_l, c_r = self.c_l.repeat(batch, 1), self.c_r.repeat(batch, 1)
        for l_i in range(len(inp)):
            r_i = len(inp) - l_i
            emb_l_i, emb_r_i = emb_ls[l_i], emb_rs[r_i]
            c_l = self.act(self.W_l(c_l) + emb_l_i)
            c_ls[l_i, :, :] = c_l
            c_r = self.act(self.W_r(c_r) + emb_r_i)
            c_rs[r_i, :, :] = c_r

        # - project recursive embeddings
        cat = torch.cat([c_ls, emb, c_rs], dimension=2) \
                   .view(seq_len * batch, self.cat_dim)
        y = self.act(self.word_proj(cat))
        # - element-wise argmax over recursive embeddings
        y_max = torch.max(y, 0)[0].squeeze(0)  # (batch_size x cat_dim)
        return F.softmax(self.doc_proj(y_max))  # (batch_size x n_classes)
