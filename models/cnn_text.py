
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseTextNN


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


class CNNText(BaseTextNN):
    """
    Implementation of 'Convolutional Neural Networks for Sentence Classification'
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
                 out_channels=128, kernel_sizes=(5, 4, 3), act='relu',
                 # rest parameters
                 dropout=0.0, **kwargs):
        self.vocab = vocab
        self.emb_dim = emb_dim

        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.dropout = dropout

        super(CNNText, self).__init__()

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
