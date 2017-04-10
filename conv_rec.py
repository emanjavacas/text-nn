
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 vocab, emb_dim,
                 # cnn parameters
                 out_channels, kernel_sizes,
                 # rnn parameters
                 hid_dim, cell='LSTM', num_layers=1, bidi=True,
                 # rest parameters
                 dropout=0.0, act='ReLU', padding_idx=None):
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

        super(ConvRNN, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(
            self.vocab, self.emb_dim, padding_idx=padding_idx)

        # CNN
        self.convs = []
        for kernel_size in self.kernel_sizes:
            self.convs.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels,
                          # number of filters HxW for each kernel (H is
                          # fixed to emb_dim to ensure H_out equals one)
                          (self.emb_dim, kernel_size)),
                getattr(nn, self.act)))

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
        emb = emb.transpose(1, 2).unsqueeze(1)  # (batch x 1 x emb_dim x seq_len)

        # CNN
        for conv in self.convs:
            # (batch x out_channels x 1 x W_out)
            out = conv(emb)
            # (batch x out_channels x )
            out = F.max_pool1d(out, (1, out.size(3)))
            out = F.dropout(out, p=self.dropout, training=self.training)
            emb = out.view()
        conv_out = [getattr(F, self.act)(conv(emb)) for conv in self.convs]
        # [(batch x out_channels x W_out), ...]
        conv_out = [out_i.squeeze(2) for out_i in conv_out]
        # [(batch x out_channels x 1), ...]
        pool_out = [F.max_pool1d(out_i, out_i.size(2)) for out_i in conv_out]
        # [(batch x out_channels), ...]
        pool_out = [out_i.squeeze(2) for out_i in pool_out]
        pool_out = torch.stack(pool_out)  # (num_kernels x batch x out_channels)
        pool_out = F.dropout(out, dropout=self.dropout, training=self.training)

        # RNN
        rnn_out, _ = self.rnn(pool_out)
        rnn_out = rnn_out[-1]   # take only last layer & last state output
        return F.log_softmax(self.proj(h_n)
