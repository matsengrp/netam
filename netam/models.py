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


class FivemerModel(nn.Module):
    def __init__(self, dataset):
        super(FivemerModel, self).__init__()
        self.all_kmers = dataset.all_kmers
        self.kmer_count = len(dataset.kmer_to_index)

        self.kmer_embedding = nn.Embedding(self.kmer_count, 1)

    def forward(self, encoded_parents, masks):
        log_kmer_rates = self.kmer_embedding(encoded_parents).squeeze()
        rates = torch.exp(log_kmer_rates)
        return rates

    @property
    def kmer_rates(self):
        # Convert kmer log rates to linear space
        return torch.exp(self.kmer_embedding.weight).squeeze()


class SHMoofModel(nn.Module):
    def __init__(self, dataset):
        super(SHMoofModel, self).__init__()
        self.all_kmers = dataset.all_kmers
        self.kmer_count = len(dataset.kmer_to_index)
        self.site_count = dataset.max_length

        self.kmer_embedding = nn.Embedding(self.kmer_count, 1)
        self.log_site_rates = nn.Embedding(self.site_count, 1)

    def forward(self, encoded_parents, masks):
        log_kmer_rates = self.kmer_embedding(encoded_parents).squeeze()
        sequence_length = encoded_parents.size(1)
        positions = torch.arange(sequence_length, device=encoded_parents.device)
        # When we transpose we get a tensor of shape [sequence_length, 1], which will broadcast
        # to the shape of log_kmer_rates, repeating over the batch dimension.
        log_site_rates = self.log_site_rates(positions).T
        # Rates are the product of kmer and site rates.
        rates = torch.exp(log_kmer_rates + log_site_rates)
        return rates

    @property
    def kmer_rates(self):
        # Convert kmer log rates to linear space
        return torch.exp(self.kmer_embedding.weight).squeeze()

    @property
    def site_rates(self):
        # Convert site log rates to linear space
        return torch.exp(self.log_site_rates.weight).squeeze()

    def write_shmoof_output(self, out_dir):
        # Extract k-mer (motif) mutabilities
        kmer_rates = self.kmer_rates.detach().numpy().flatten()
        motif_mutabilities = pd.DataFrame(
            {
                "Motif": self.all_kmers,
                "Mutability": kmer_rates,
            }
        )
        motif_mutabilities.to_csv(
            f"{out_dir}/motif_mutabilities.tsv", sep="\t", index=False
        )

        # Extract site mutabilities
        site_mutabilities = self.site_rates.detach().numpy().flatten()
        site_mutabilities_df = pd.DataFrame(
            {
                "Position": range(1, len(site_mutabilities) + 1),
                "Mutability": site_mutabilities,
            }
        )
        site_mutabilities_df.to_csv(
            f"{out_dir}/site_mutabilities.tsv", sep="\t", index=False
        )


class NoofModel(nn.Module):
    def __init__(
        self, dataset, embedding_dim, nhead, dim_feedforward, layer_count, dropout=0.5
    ):
        super(NoofModel, self).__init__()
        self.kmer_count = len(dataset.kmer_to_index)
        self.embedding_dim = embedding_dim
        self.site_count = dataset.max_length

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

    def forward(self, encoded_parents, masks):
        """
        The forward method.

        encoded_parents is expected to be an integer tensor of [batch_size, sequence_length].
        """
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


class CNNModel(nn.Module):
    def __init__(self, dataset, embedding_dim, num_filters, kernel_size, dropout_rate=0.1):
        super(CNNModel, self).__init__()
        self.kmer_count = len(dataset.kmer_to_index)
        self.kmer_embedding = nn.Embedding(self.kmer_count, embedding_dim)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size, padding='same')
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(in_features=num_filters, out_features=1)

    def forward(self, encoded_parents, masks):
        kmer_embeds = self.kmer_embedding(encoded_parents)
        kmer_embeds = kmer_embeds.permute(0, 2, 1)  # Transpose for Conv1D
        conv_out = F.relu(self.conv(kmer_embeds))
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)  # Transpose back for Linear layer
        log_rates = self.linear(conv_out).squeeze(-1)
        rates = torch.exp(log_rates * masks)
        return rates
