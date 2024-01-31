from abc import ABC, abstractmethod
import math

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from netam.common import (
    MAX_AMBIG_AA_IDX,
    aa_idx_tensor_of_str_ambig,
    PositionalEncoding,
    generate_kmers,
    mask_tensor_of,
)


class ModelBase(nn.Module):
    def reinitialize_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Embedding):
                nn.init.normal_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.TransformerEncoder):
                for sublayer in layer.modules():
                    if isinstance(sublayer, nn.Linear):
                        nn.init.kaiming_normal_(sublayer.weight, nonlinearity="relu")
                        if sublayer.bias is not None:
                            nn.init.constant_(sublayer.bias, 0)
            elif isinstance(layer, nn.Dropout):
                pass
            elif hasattr(layer, "reinitialize_weights"):
                layer.reinitialize_weights()
            else:
                raise ValueError(f"Unrecognized layer type: {type(layer)}")


class KmerModel(ModelBase):
    def __init__(self, kmer_length):
        super().__init__()
        self.kmer_length = kmer_length
        self.all_kmers = generate_kmers(kmer_length)
        self.kmer_count = len(self.all_kmers)

    @property
    def hyperparameters(self):
        return {
            "kmer_length": self.kmer_length,
        }


class FivemerModel(KmerModel):
    def __init__(self):
        super().__init__(kmer_length=5)
        self.kmer_embedding = nn.Embedding(self.kmer_count, 1)

    def forward(self, encoded_parents, masks):
        log_kmer_rates = self.kmer_embedding(encoded_parents).squeeze(-1)
        rates = torch.exp(log_kmer_rates)
        return rates

    @property
    def kmer_rates(self):
        return torch.exp(self.kmer_embedding.weight).squeeze()


class SHMoofModel(KmerModel):
    def __init__(self, kmer_length, site_count):
        super().__init__(kmer_length)
        self.site_count = site_count
        self.kmer_embedding = nn.Embedding(self.kmer_count, 1)
        self.log_site_rates = nn.Embedding(self.site_count, 1)

    def forward(self, encoded_parents, masks):
        log_kmer_rates = self.kmer_embedding(encoded_parents).squeeze(-1)
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
        return torch.exp(self.kmer_embedding.weight).squeeze()

    @property
    def site_rates(self):
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
        self, kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob=0.1
    ):
        super().__init__(kmer_length)
        self.kmer_embedding = nn.Embedding(self.kmer_count, embedding_dim)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filter_count,
            kernel_size=kernel_size,
            padding="same",
        )
        self.dropout = nn.Dropout(dropout_prob)
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
            "dropout_prob": self.dropout.p,
        }


class CNNPEModel(CNNModel):
    def __init__(
        self, kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob
    ):
        super().__init__(
            kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob
        )
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=dropout_prob)

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

    def __init__(self, filter_count, kernel_size, dropout_prob=0.1):
        # Fixed embedding_dim because there are only 4 bases.
        embedding_dim = 5
        kmer_length = 1
        super().__init__(
            kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob
        )
        # Here's how we adapt the model to use individual bases as input rather
        # than trainable kmer embeddings.
        identity_matrix = torch.eye(embedding_dim)
        self.kmer_embedding.weight = nn.Parameter(identity_matrix, requires_grad=False)


class RSCNNModel(CNNModel):
    """
    This is a CNN model that uses k-mers as input and trains an embedding layer.
    """

    def __init__(
        self, kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob=0.1
    ):
        super().__init__(
            kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob
        )
        self.kmer_embedding = nn.Embedding(self.kmer_count, embedding_dim)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filter_count,
            kernel_size=kernel_size,
            padding="same",
        )
        self.dropout = nn.Dropout(dropout_prob)
        # Rate linear layer
        self.r_linear = nn.Linear(in_features=filter_count, out_features=1)
        # Substitution probability linear layer
        self.s_linear = nn.Linear(in_features=filter_count, out_features=4)

    def forward(self, encoded_parents, masks, wt_base_multiplier):
        kmer_embeds = self.kmer_embedding(encoded_parents)
        kmer_embeds = kmer_embeds.permute(0, 2, 1)  # Transpose for Conv1D
        conv_out = F.relu(self.conv(kmer_embeds))
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(
            0, 2, 1
        )  # Transpose back for applying linear layers

        log_rates = self.r_linear(conv_out).squeeze(-1)
        csp_raw = self.s_linear(conv_out)
        csp_raw *= wt_base_multiplier

        csp = F.softmax(csp_raw, dim=-1) * masks.unsqueeze(-1)

        rates = torch.exp(log_rates * masks)
        return rates, csp


# Issue #8
class WrapperHyperparameters:
    def __init__(self, base_model_hyperparameters, site_count):
        self.base_model_hyperparameters = base_model_hyperparameters
        self.site_count = site_count

    def __getitem__(self, key):
        if key in self.base_model_hyperparameters:
            return self.base_model_hyperparameters[key]
        elif hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found in hyperparameters.")

    def __str__(self):
        hyperparameters_dict = {key: getattr(self, key) for key in self.__dict__}
        return str(hyperparameters_dict)


class PersiteWrapper(ModelBase):
    """
    This wraps another model, but adds a per-site rate component.
    """

    def __init__(self, base_model, site_count):
        super().__init__()
        self.base_model = base_model
        self.site_count = site_count
        self.log_site_rates = nn.Embedding(self.site_count, 1)
        self._hyperparameters = WrapperHyperparameters(
            self.base_model.hyperparameters, self.site_count
        )

    def forward(self, encoded_parents, masks):
        base_model_rates = self.base_model(encoded_parents, masks)
        sequence_length = encoded_parents.size(1)
        positions = torch.arange(sequence_length, device=encoded_parents.device)
        log_site_rates = self.log_site_rates(positions).T
        rates = base_model_rates * torch.exp(log_site_rates)
        return rates

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @property
    def site_rates(self):
        return torch.exp(self.log_site_rates.weight).squeeze()


class AbstractBinarySelectionModel(ABC, nn.Module):
    """A transformer-based model for binary selection.

    This is a model that takes in a batch of one-hot encoded sequences and
    outputs a vector that represents the log level of selection for each amino
    acid site, which after exponentiating is a multiplier on the probability of
    an amino-acid substitution at that site.

    Various submodels are implemented as subclasses of this class:

    * LinAct: No activation function after the transformer.
    * WiggleAct: Activation that slopes to 0 at -inf and grows sub-linearly as x increases.

    See forward() for details.
    """

    def __init__(self):
        super().__init__()

    def selection_factors_of_aa_str(self, aa_str: str) -> Tensor:
        """Do the forward method without gradients from an amino acid string.

        Parameters:
            aa_str: A string of amino acids.

        Returns:
            A numpy array of the same length as the input string representing
            the level of selection for wildtype at each amino acid site.
        """

        model_device = next(self.parameters()).device

        aa_idxs = aa_idx_tensor_of_str_ambig(aa_str)
        aa_idxs = aa_idxs.to(model_device)
        mask = mask_tensor_of(aa_str)
        mask = mask.to(model_device)

        with torch.no_grad():
            model_out = self(aa_idxs.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
            final_out = torch.exp(model_out)

        return final_out[: len(aa_str)]


class TransformerBinarySelectionModelLinAct(AbstractBinarySelectionModel):
    def __init__(
        self,
        nhead: int,
        d_model_per_head: int,
        dim_feedforward: int,
        layer_count: int,
        dropout_prob: float = 0.5,
    ):
        super().__init__()
        # Note that d_model has to be divisible by nhead, so we make that
        # automatic here.
        self.d_model_per_head = d_model_per_head
        self.d_model = d_model_per_head * nhead
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.pos_encoder = PositionalEncoding(self.d_model, dropout_prob)
        self.amino_acid_embedding = nn.Embedding(MAX_AMBIG_AA_IDX + 1, self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layer_count)
        self.linear = nn.Linear(self.d_model, 1)
        self.init_weights()

    @property
    def hyperparameters(self):
        return {
            "nhead": self.nhead,
            "d_model_per_head": self.d_model_per_head,
            "dim_feedforward": self.dim_feedforward,
            "layer_count": self.encoder.num_layers,
            "dropout_prob": self.pos_encoder.dropout.p,
        }

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, amino_acid_indices: Tensor, mask: Tensor) -> Tensor:
        """Build a binary log selection matrix from a one-hot encoded parent sequence.

        Because we're predicting log of the selection factor, we don't use an
        activation function after the transformer.

        Parameters:
            amino_acid_indices: A tensor of shape (B, L) containing the indices of parent AA sequences.
            mask: A tensor of shape (B, L) representing the mask of valid amino acid sites.

        Returns:
            A tensor of shape (B, L, 1) representing the log level of selection
            for each amino acid site.
        """
        # Multiply by sqrt(d_model) to match the transformer paper.
        embedded_amino_acids = self.amino_acid_embedding(
            amino_acid_indices
        ) * math.sqrt(self.d_model)
        # Have to do the permutation because the positional encoding expects the
        # sequence length to be the first dimension.
        embedded_amino_acids = self.pos_encoder(
            embedded_amino_acids.permute(1, 0, 2)
        ).permute(1, 0, 2)

        # To learn about src_key_padding_mask, see https://stackoverflow.com/q/62170439
        out = self.encoder(embedded_amino_acids, src_key_padding_mask=~mask)
        out = self.linear(out)
        return out.squeeze(-1)


def wiggle(x, beta):
    """
    A function that when we exp it gives us a function that slopes to 0 at -inf
    and grows sub-linearly as x increases.

    See https://github.com/matsengrp/netam/pull/5#issuecomment-1906665475 for a
    plot.
    """
    return beta * torch.where(x < 1, x - 1, torch.log(x))


class TransformerBinarySelectionModelWiggleAct(TransformerBinarySelectionModelLinAct):
    """
    Here the beta parameter is fixed at 0.3.
    """

    def forward(self, amino_acid_indices: Tensor, mask: Tensor) -> Tensor:
        return wiggle(super().forward(amino_acid_indices, mask), 0.3)


class TransformerBinarySelectionModelTrainableWiggleAct(
    TransformerBinarySelectionModelLinAct
):
    """
    This version of the model has a trainable parameter that controls the
    beta in the wiggle function. It didn't work any better so I'm not using it
    for now.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the logit of beta to logit(0.3)
        init_beta = 0.3
        init_logit_beta = math.log(init_beta / (1 - init_beta))
        self.logit_beta = nn.Parameter(
            torch.tensor([init_logit_beta], dtype=torch.float32)
        )

    def forward(self, amino_acid_indices: Tensor, mask: Tensor) -> Tensor:
        # Apply sigmoid to transform logit_beta back to the range (0, 1)
        beta = torch.sigmoid(self.logit_beta)
        return wiggle(super().forward(amino_acid_indices, mask), beta)


class SingleValueBinarySelectionModel(AbstractBinarySelectionModel):
    """A one parameter selection model as a baseline."""

    def __init__(self):
        super().__init__()
        self.single_value = nn.Parameter(torch.tensor(0.0))

    @property
    def hyperparameters(self):
        return {}

    def forward(self, amino_acid_indices: Tensor, mask: Tensor) -> Tensor:
        """Build a binary log selection matrix from a one-hot encoded parent sequence."""
        replicated_value = self.single_value.expand_as(amino_acid_indices)
        return replicated_value
