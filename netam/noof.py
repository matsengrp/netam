import itertools

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from epam.torch_common import PositionalEncoding

BASES = ["A", "C", "G", "T"]


class NoofModel(nn.Module):
    def __init__(
        self, dataset, embedding_dim, nhead, dim_feedforward, layer_count, dropout=0.5
    ):
        super(NoofModel, self).__init__()
        self.kmer_count = len(dataset.kmer_to_index)
        self.embedding_dim = embedding_dim
        self.site_count = dataset.max_length

        # self.kmer_embedding = nn.Embedding(self.kmer_count, 1)
        self.kmer_embedding = nn.Embedding(self.kmer_count, self.embedding_dim)
        self.pos_encoder = PositionalEncoding(self.embedding_dim, dropout)

        self.encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = TransformerEncoder(self.encoder_layer, layer_count)
        self.linear = nn.Linear(self.embedding_dim, 1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, encoded_parents):
        """
        The forward method.

        encoded_parents is expected to be an integer tensor of [batch_size, sequence_length].
        """
        # log_rates = self.kmer_embedding(encoded_parents).squeeze()
        kmer_embeddings = self.kmer_embedding(encoded_parents)
        kmer_embeddings = self.pos_encoder(kmer_embeddings)

        # Pass through the transformer encoder
        transformer_output = self.encoder(kmer_embeddings)

        # Apply the linear layer and squeeze out the last dimension.
        # After the linear layer, the dimensions will be [batch_size, sequence_length, 1].
        # We squeeze out the last dimension to make it [batch_size, sequence_length].
        log_rates = self.linear(transformer_output).squeeze(-1)
        rates = torch.exp(log_rates)
        return rates


