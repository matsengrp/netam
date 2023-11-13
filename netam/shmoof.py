import itertools

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import itertools

BASES = ["A", "C", "G", "T"]


def generate_kmers_with_padding(kmer_length, max_padding_length):
    """
    Generate all possible k-mers of a specified length composed of 'A', 'C', 'G', 'T',
    with allowances for up to `max_padding_length` 'N's at either the beginning or end
    of the k-mer.

    Parameters
    ----------
    kmer_length : int
        The length of the k-mers to be generated.
    max_padding_length : int
        The maximum number of 'N' nucleotides allowed as padding at the beginning or
        end of the k-mer.

    Returns
    -------
    list of str
        A sorted list of unique k-mers. Each k-mer is a string of length `kmer_length`,
        composed of 'A', 'C', 'G', 'T', and up to `max_padding_length` 'N's at either
        the beginning or end of the k-mer. The list is generated in lexicographic order
        for the unpadded k-mers and then all of the padded k-mers are appended to the
        end of the list.
    """

    # Add all combinations of ACGT
    all_kmers = ["".join(p) for p in itertools.product(BASES, repeat=kmer_length)]

    # Add combinations with up to max_padding_length Ns at the beginning or end
    for n in range(1, max_padding_length + 1):
        for kmer in itertools.product(BASES, repeat=kmer_length - n):
            kmer_str = "".join(kmer)
            all_kmers.append("N" * n + kmer_str)
            all_kmers.append(kmer_str + "N" * n)

    return all_kmers


class SHMoofDataset(Dataset):
    def __init__(self, dataframe, kmer_length, max_length):
        self.max_length = max_length
        self.kmer_length = kmer_length
        self.overhang_length = (kmer_length - 1) // 2
        assert self.overhang_length > 0 and kmer_length % 2 == 1

        self.all_kmers = generate_kmers_with_padding(kmer_length, self.overhang_length)
        assert len(self.all_kmers) < torch.iinfo(torch.int32).max
        self.resolved_kmer_count = 4**kmer_length
        for resolved_kmers in self.all_kmers[: self.resolved_kmer_count]:
            assert "N" not in resolved_kmers
        for padded_kmers in self.all_kmers[self.resolved_kmer_count :]:
            assert "N" in padded_kmers

        self.kmer_to_index = {kmer: idx for idx, kmer in enumerate(self.all_kmers)}

        (
            self.encoded_parents,
            self.masks,
            self.mutation_vectors,
        ) = self.encode_sequences(dataframe)

    @property
    def resolved_kmers(self):
        return self.all_kmers[: self.resolved_kmer_count]

    def __len__(self):
        return len(self.encoded_parents)

    def __getitem__(self, idx):
        return self.encoded_parents[idx], self.masks[idx], self.mutation_vectors[idx]

    def encode_sequences(self, dataframe):
        encoded_parents = []
        masks = []
        mutation_vectors = []

        for _, row in dataframe.iterrows():
            encoded_parent, mask = self.encode_sequence(row["parent"])
            mutation_indicator = self.create_mutation_indicator(
                row["parent"], row["child"]
            )

            encoded_parents.append(encoded_parent)
            masks.append(mask)
            mutation_vectors.append(mutation_indicator)

        return (
            torch.stack(encoded_parents),
            torch.stack(masks),
            torch.stack(mutation_vectors),
        )

    def encode_sequence(self, sequence):
        # Pad sequence with overhang_length 'N's at the start and end so that we
        # can assign parameters to every site in the sequence.
        padded_sequence = (
            "N" * self.overhang_length + sequence + "N" * self.overhang_length
        )
        # Truncate according to max_length; we'll handle the overhangs later.
        padded_sequence = padded_sequence[: self.max_length + 2 * self.overhang_length]

        # Here we use a default value of 0, which wouldn't be a good idea except for
        # the fact that we do masking to ignore these values.
        kmer_indices = [
            self.kmer_to_index.get(padded_sequence[i : i + self.kmer_length], 0)
            for i in range(self.max_length)
        ]

        mask = [1 if i < len(sequence) else 0 for i in range(self.max_length)]

        return torch.tensor(kmer_indices, dtype=torch.int32), torch.tensor(
            mask, dtype=torch.bool
        )

    def create_mutation_indicator(self, parent, child):
        assert len(parent) == len(
            child
        ), f"{parent} and {child} are not the same length"
        mutation_indicator = [
            1 if parent[i] != child[i] else 0
            for i in range(min(len(parent), self.max_length))
        ]

        # Pad the mutation indicator if necessary
        if len(mutation_indicator) < self.max_length:
            mutation_indicator += [0] * (self.max_length - len(mutation_indicator))

        return torch.tensor(mutation_indicator, dtype=torch.bool)


class SHMoofModel(nn.Module):
    def __init__(self, dataset, embedding_dim):
        super(SHMoofModel, self).__init__()
        self.kmer_to_index = dataset.kmer_to_index
        self.kmer_count = len(dataset.kmer_to_index)
        self.embedding_dim = embedding_dim
        self.site_count = dataset.max_length

        # Limit embedding size to the number of fully specified k-mers
        self.kmer_embedding = nn.Embedding(
            dataset.resolved_kmer_count, self.embedding_dim
        )
        self.padded_kmer_to_kmer_index_map = (
            self.generate_padded_kmer_to_kmer_index_map()
        )

        self.log_site_rates = nn.Embedding(self.site_count, 1)

    def generate_padded_kmer_to_kmer_index_map(self):
        """Precompute the indices of all k-mers that can be substituted for each k-mer."""
        mapping = {}
        for kmer, idx in self.kmer_to_index.items():
            if "N" in kmer:
                resolved_kmers = self.resolve_padded_kmer(kmer)
                resolved_kmer_indices = [
                    self.kmer_to_index[sub] for sub in resolved_kmers
                ]
            else:
                resolved_kmer_indices = [idx]
            mapping[idx] = resolved_kmer_indices
        return mapping

    def resolve_padded_kmer(self, kmer):
        """Find all possible resolved kmers for a kmer with some N padding."""
        resolved_kmers = []
        for replacement_bases in itertools.product("ACGT", repeat=kmer.count("N")):
            temp_kmer = kmer
            # Replace the N's, one at a time.
            for base in replacement_bases:
                temp_kmer = temp_kmer.replace("N", base, 1)
            resolved_kmers.append(temp_kmer)
        return resolved_kmers

    def get_log_kmer_rates(self, encoded_parent):
        log_kmer_rates = torch.empty((encoded_parent.size(0), self.embedding_dim))

        for i, idx in enumerate(encoded_parent):
            kmer_indices = self.padded_kmer_to_kmer_index_map[idx.item()]
            averaged_log_rates = self.kmer_embedding(
                torch.tensor(kmer_indices, dtype=torch.int32)
            ).mean(dim=0)
            log_kmer_rates[i] = averaged_log_rates

        return log_kmer_rates.squeeze()

    def forward(self, encoded_parent):
        log_kmer_rates = self.get_log_kmer_rates(encoded_parent).squeeze()
        positions = torch.arange(encoded_parent.size(0), device=encoded_parent.device)
        log_site_rates = self.log_site_rates(positions).squeeze()

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


class SHMoofBurrito:
    def __init__(self, train_dataframe, val_dataframe, kmer_length=5, max_length=300):
        self.train_dataset = SHMoofDataset(train_dataframe, kmer_length, max_length)
        self.val_dataset = SHMoofDataset(val_dataframe, kmer_length, max_length)

        self.model = SHMoofModel(self.train_dataset, embedding_dim=1)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epochs):
        self.model.train()  # Set the model to training mode
        for epoch in range(epochs):
            running_loss = 0.0
            for encoded_parent, mask, mutation_indicator in self.train_dataset:
                loss = self._calculate_loss(encoded_parent, mask, mutation_indicator)

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Average loss for this epoch
            epoch_loss = running_loss / len(self.train_dataset)

            # Validation phase
            self.model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for encoded_parent, mask, mutation_indicator in self.val_dataset:
                    loss = self._calculate_loss(
                        encoded_parent, mask, mutation_indicator
                    )
                    validation_loss += loss.item()
            validation_loss /= len(self.val_dataset)
            print(
                f"Epoch [{epoch+1}/{epochs}]\t Loss: {epoch_loss:.8f}\t Val Loss: {validation_loss:.8f}"
            )

    def _calculate_loss(self, encoded_parent, mask, mutation_indicator):
        rates = self.model(encoded_parent)
        mutation_freq = (mutation_indicator / mask.sum()).sum()
        mut_prob = 1 - torch.exp(-rates * mutation_freq)
        mut_prob_masked = mut_prob[mask]
        mutation_indicator_masked = mutation_indicator[mask].float()
        return self.criterion(mut_prob_masked, mutation_indicator_masked)

    def write_shmoof_output(self, out_dir):
        # Extract k-mer (motif) mutabilities
        kmer_mutabilities = self.model.kmer_rates.detach().numpy().flatten()
        motif_mutabilities = pd.DataFrame(
            {
                "Motif": self.train_dataset.resolved_kmers,
                "Mutability": kmer_mutabilities,
            }
        )
        motif_mutabilities.to_csv(
            f"{out_dir}/motif_mutabilities.tsv", sep="\t", index=False
        )

        # Extract site mutabilities
        site_mutabilities = self.model.site_rates.detach().numpy().flatten()
        site_mutabilities_df = pd.DataFrame(
            {
                "Position": range(1, len(site_mutabilities) + 1),
                "Mutability": site_mutabilities,
            }
        )
        site_mutabilities_df.to_csv(
            f"{out_dir}/site_mutabilities.tsv", sep="\t", index=False
        )
