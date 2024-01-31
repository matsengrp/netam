import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from netam.common import BASES_AND_N_TO_INDEX
from netam.framework import SHMBurrito
from netam.models import KmerModel


class RSCNNModel(KmerModel):
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


class RSSHMBurrito(SHMBurrito):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xent_loss = nn.CrossEntropyLoss()

    def loss_of_batch(self, batch):
        # TODO consider a weighted sum of losses, see README
        encoded_parents, masks, mutation_indicators, new_base_idxs, wt_base_multiplier = batch
        rates, csp = self.model(encoded_parents, masks, wt_base_multiplier)

        # Existing mutation rate loss calculation
        mutation_freq = mutation_indicators.sum(dim=1, keepdim=True) / masks.sum(
            dim=1, keepdim=True
        )
        mut_prob = 1 - torch.exp(-rates * mutation_freq)
        mut_prob_masked = mut_prob[masks]
        mutation_indicator_masked = mutation_indicators[masks].float()
        rate_loss = self.bce_loss(mut_prob_masked, mutation_indicator_masked)

        # Conditional substitution probability (CSP) loss calculation
        # Mask the new_base_idxs to focus only on positions with mutations
        mutated_positions_mask = mutation_indicators == 1
        csp_masked = csp[mutated_positions_mask]
        new_base_idxs_masked = new_base_idxs[mutated_positions_mask]
        assert (new_base_idxs_masked >= 0).all()

        csp_loss = self.xent_loss(csp_masked, new_base_idxs_masked)

        total_loss = rate_loss + csp_loss

        return total_loss

def burrito_class_of_model(model):
    if isinstance(model, RSCNNModel):
        return RSSHMBurrito
    else:
        return SHMBurrito