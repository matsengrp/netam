import itertools

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


import torch
import itertools
from torch.utils.data import Dataset


class SHMoofDataset(Dataset):
    def __init__(self, dataframe, max_length, kmer_length=5):
        self.max_length = max_length
        self.kmer_length = kmer_length
        self.overhang_length = (kmer_length - 1) // 2
        assert self.overhang_length > 0 and kmer_length % 2 == 1

        self.all_kmers = [
            "".join(p) for p in itertools.product("ACGTN", repeat=kmer_length)
        ]
        assert len(self.all_kmers) < torch.iinfo(torch.int32).max

        self.kmer_to_index = {kmer: idx for idx, kmer in enumerate(self.all_kmers)}

        (
            self.encoded_parents,
            self.masks,
            self.mutation_vectors,
        ) = self.encode_sequences(dataframe)

    def __getitem__(self, idx):
        return self.encoded_parents[idx], self.masks[idx], self.mutation_vectors[idx]

    def encode_sequences(self, dataframe):
        encoded_parents = []
        masks = []
        mutation_vectors = []

        for _, row in dataframe.iterrows():
            encoded_parent, mask = self.encode_sequence(row["parent"])
            mutation_vector = self.create_mutation_vector(row["parent"], row["child"])

            encoded_parents.append(encoded_parent)
            masks.append(mask)
            mutation_vectors.append(mutation_vector)

        return (
            torch.stack(encoded_parents),
            torch.stack(masks),
            torch.stack(mutation_vectors),
        )


    def encode_sequence(self, sequence):
        # Pad sequence with overhang_length 'N's at the start and end
        padded_sequence = 'N' * self.overhang_length + sequence + 'N' * self.overhang_length
        padded_sequence = padded_sequence[:self.max_length + 2 * self.overhang_length]

        kmer_indices = [
            self.kmer_to_index.get(padded_sequence[i : i + self.kmer_length], 0)
            for i in range(self.max_length)
        ]

        mask = [1 if i < len(sequence) else 0 for i in range(self.max_length)]

        return torch.tensor(kmer_indices, dtype=torch.int32), torch.tensor(mask, dtype=torch.bool)


    def create_mutation_vector(self, parent, child):
        mutation_vector = [
            1 if parent[i] != child[i] and i < self.max_length else 0
            for i in range(len(parent))
        ]
        mutation_vector += [0] * (self.max_length - len(mutation_vector))
        return torch.tensor(mutation_vector, dtype=torch.bool)


class SHMoofModel(nn.Module):
    def __init__(self, dataset):
        super(SHMoofModel, self).__init__()
        # Extract counts from the dataset
        self.kmer_count = len(dataset.kmer_to_index)
        self.site_count = dataset.max_length

        # Initialize embedding layers
        self.kmer_rates = nn.Embedding(self.kmer_count, 1)
        self.site_rates = nn.Embedding(self.site_count, 1)

    def forward(self, encoded_parent):
        # Get kmer rates and site rates, assuming both are 1D tensors
        kmer_rates = self.kmer_rates(encoded_parent).squeeze()
        positions = torch.arange(encoded_parent.size(0), device=encoded_parent.device)
        site_rates = self.site_rates(positions).squeeze()

        rates = kmer_rates + site_rates

        return rates
