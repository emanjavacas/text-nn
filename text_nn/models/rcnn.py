
import torch
import torch.nn as nn
import torch.nn.functional as F

from text_nn.models.base import BaseTextNN


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
        y = self.act(self.max_proj(cat)) \
                .view(seq_len, batch, self.max_dim)
        y = F.dropout(y, p=self.dropout, training=self.training)

        # - element-wise argmax over concatenated contexts
        y_max = torch.max(y, 0)[0].squeeze(0)  # (batch_size x cat_dim)
        return F.log_softmax(self.doc_proj(y_max))  # (batch_size x n_classes)
