import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from netam.common import BASES_AND_N
from netam.models import KmerModel

class RSCNNModel(KmerModel):
    """
    This is a CNN model that uses k-mers as input and trains an embedding layer.
    """
    def __init__(self, kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob=0.1):
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

        self.central_base_mapping = torch.tensor([BASES_AND_N.index(kmer[len(kmer)//2]) for kmer in self.all_kmers], dtype=torch.int64)


    def forward(self, encoded_parents, masks):
        kmer_embeds = self.kmer_embedding(encoded_parents)
        kmer_embeds = kmer_embeds.permute(0, 2, 1)  # Transpose for Conv1D
        conv_out = F.relu(self.conv(kmer_embeds))
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)  # Transpose back for applying linear layers

        log_rates = self.r_linear(conv_out).squeeze(-1)
        csp_raw = self.s_linear(conv_out)
        
        # Use the kmer indices to get the central base indices
        central_bases = self.central_base_mapping[encoded_parents]

        batch_size, seq_length, _ = csp_raw.size()
        for batch in range(batch_size):
            for i in range(seq_length):
                base_idx = central_bases[batch, i]
                if base_idx < 4:  # Skip if the central base is N
                    csp_raw[batch, i, base_idx] = -float('inf')

        csp = F.softmax(csp_raw, dim=-1) * masks.unsqueeze(-1)

        rates = torch.exp(log_rates * masks)
        return rates, csp