import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from netam.common import generate_kmers, PositionalEncoding


class KmerModel(nn.Module):
    def __init__(self, kmer_length):
        super(KmerModel, self).__init__()
        self.kmer_length = kmer_length
        self.all_kmers = generate_kmers(kmer_length)
        self.kmer_count = len(self.all_kmers)

    @property
    def hyperparameters(self):
        return {
            "kmer_length": self.kmer_length,
        }


class FivemerModel(KmerModel):
    def __init__(self, kmer_length):
        super(FivemerModel, self).__init__(kmer_length)
        self.kmer_embedding = nn.Embedding(self.kmer_count, 1)

    def forward(self, encoded_parents, masks):
        log_kmer_rates = self.kmer_embedding(encoded_parents).squeeze()
        rates = torch.exp(log_kmer_rates)
        return rates

    @property
    def kmer_rates(self):
        # Convert kmer log rates to linear space
        return torch.exp(self.kmer_embedding.weight).squeeze()


class SHMoofModel(KmerModel):
    def __init__(self, kmer_length, site_count):
        super(SHMoofModel, self).__init__(kmer_length)
        self.site_count = site_count
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
    def hyperparameters(self):
        return {
            "kmer_length": self.kmer_length,
            "site_count": self.site_count,
        }

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


class CNNModel(KmerModel):
    """
    This is a CNN model that uses k-mers as input and trains an embedding layer.
    """

    def __init__(
        self, kmer_length, embedding_dim, filter_count, kernel_size, dropout_rate=0.1
    ):
        super(CNNModel, self).__init__(kmer_length)
        self.kmer_embedding = nn.Embedding(self.kmer_count, embedding_dim)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filter_count,
            kernel_size=kernel_size,
            padding="same",
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(in_features=filter_count, out_features=1)

    def forward(self, encoded_parents, masks):
        kmer_embeds = self.kmer_embedding(encoded_parents)
        kmer_embeds = kmer_embeds.permute(0, 2, 1)  # Transpose for Conv1D
        conv_out = F.relu(self.conv(kmer_embeds))
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)  # Transpose back for Linear layer
        log_rates = self.linear(conv_out).squeeze(-1)
        rates = torch.exp(log_rates * masks)
        return rates

    @property
    def hyperparameters(self):
        return {
            "kmer_length": self.kmer_length,
            "embedding_dim": self.kmer_embedding.embedding_dim,
            "filter_count": self.conv.out_channels,
            "kernel_size": self.conv.kernel_size[0],
            "dropout_rate": self.dropout.p,
        }


class CNNPEModel(CNNModel):
    def __init__(self, kmer_length, embedding_dim, filter_count, kernel_size, dropout_rate):
        super(CNNModel, self).__init__(self, kmer_length, embedding_dim, filter_count, kernel_size, dropout_rate)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=dropout_rate)

    def forward(self, encoded_parents, masks):
        kmer_embeds = self.kmer_embedding(encoded_parents)
        kmer_embeds = self.pos_encoder(kmer_embeds)
        kmer_embeds = kmer_embeds.permute(0, 2, 1)  # Transpose for Conv1D
        conv_out = F.relu(self.conv(kmer_embeds))
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)  # Transpose back for Linear layer
        log_rates = self.linear(conv_out).squeeze(-1)
        rates = torch.exp(log_rates * masks)
        return rates


class CNN1merModel(CNNModel):
    """
    This is a CNN model that uses individual bases as input and does not train an
    embedding layer.
    """

    def __init__(self, filter_count, kernel_size, dropout_rate=0.1):
        # Fixed embedding_dim because there are only 4 bases.
        embedding_dim = 5
        kmer_length = 1
        super(CNN1merModel, self).__init__(
            kmer_length, embedding_dim, filter_count, kernel_size, dropout_rate
        )
        identity_matrix = torch.eye(embedding_dim)
        self.kmer_embedding.weight = nn.Parameter(identity_matrix, requires_grad=False)


class PersiteWrapper(nn.Module):
    """
    This wraps another model, but adds a per-site rate component.
    """

    def __init__(self, base_model, site_count):
        super(PersiteWrapper, self).__init__()
        self.base_model = base_model
        self.site_count = site_count
        self.log_site_rates = nn.Embedding(self.site_count, 1)

    def forward(self, encoded_parents, masks):
        base_model_rates = self.base_model(encoded_parents, masks)
        sequence_length = encoded_parents.size(1)
        positions = torch.arange(sequence_length, device=encoded_parents.device)
        log_site_rates = self.log_site_rates(positions).T
        rates = base_model_rates * torch.exp(log_site_rates)
        return rates

    @property
    def hyperparameters(self):
        return {
            "base_model_hyperparameters": self.base_model.hyperparameters,
            "site_count": self.site_count,
        }

    @property
    def site_rates(self):
        # Convert site log rates to linear space
        return torch.exp(self.log_site_rates.weight).squeeze()
