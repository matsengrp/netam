import itertools

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader

from epam.torch_common import PositionalEncoding

BASES = ["A", "C", "G", "T"]


class SHMoofDataset(Dataset):
    def __init__(self, dataframe, kmer_length, max_length):
        self.max_length = max_length
        self.kmer_length = kmer_length
        self.overhang_length = (kmer_length - 1) // 2
        assert self.overhang_length > 0 and kmer_length % 2 == 1

        # Our strategy to kmers is to have a single representation for any kmer that isn't in ACGT.
        # This is the first one so is the default value below.
        self.all_kmers = ["N"] + [
            "".join(p) for p in itertools.product(BASES, repeat=kmer_length)
        ]
        assert len(self.all_kmers) < torch.iinfo(torch.int32).max
        self.kmer_to_index = {kmer: idx for idx, kmer in enumerate(self.all_kmers)}

        (
            self.encoded_parents,
            self.masks,
            self.mutation_vectors,
        ) = self.encode_sequences(dataframe)

    def __len__(self):
        return len(self.encoded_parents)

    def __getitem__(self, idx):
        return self.encoded_parents[idx], self.masks[idx], self.mutation_vectors[idx]

    def to(self, device):
        self.encoded_parents = self.encoded_parents.to(device)
        self.masks = self.masks.to(device)
        self.mutation_vectors = self.mutation_vectors.to(device)

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

        # Note that we are using a default value of 0 here. So we use the
        # catch-all term for anything with an N in it for the sites on the
        # boundary of the kmer. 
        # Note that this line also effectively pads things out to max_length because
        # when i gets large the slice will be empty and we will get a 0.
        # These sites will get masked out by the mask below.
        kmer_indices = [
            self.kmer_to_index.get(padded_sequence[i : i + self.kmer_length], 0)
            for i in range(self.max_length)
        ]

        mask = [
            1 if i < len(sequence) and sequence[i] != "N" else 0
            for i in range(self.max_length)
        ]

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
    def __init__(self, dataset):
        super(SHMoofModel, self).__init__()
        self.kmer_count = len(dataset.kmer_to_index)
        self.site_count = dataset.max_length

        self.kmer_embedding = nn.Embedding(self.kmer_count, 1)
        self.log_site_rates = nn.Embedding(self.site_count, 1)

# full batch version
#     def forward(self, encoded_parents):
        # log_kmer_rates = self.kmer_embedding(encoded_parents).squeeze()
        # sequence_length = encoded_parents.size(1)
        # positions = torch.arange(sequence_length, device=encoded_parents.device)
        # # When we transpose we get a tensor of shape [sequence_length, 1], which will broadcast
        # # to the shape of log_kmer_rates, repeating over the batch dimension.
        # log_site_rates = self.log_site_rates(positions).T
#         # Rates are the product of kmer and site rates.
#         rates = torch.exp(log_kmer_rates + log_site_rates)
#         return rates

# fake batch version
    def forward(self, encoded_parent):
        # fake batch dimension
        encoded_parents = encoded_parent.unsqueeze(0)
        log_kmer_rates = self.kmer_embedding(encoded_parents).squeeze()
        sequence_length = encoded_parents.size(1)
        positions = torch.arange(sequence_length, device=encoded_parents.device)
        # When we transpose we get a tensor of shape [sequence_length, 1], which will broadcast
        # to the shape of log_kmer_rates, repeating over the batch dimension.
        log_site_rates = self.log_site_rates(positions).T
        # Rates are the product of kmer and site rates.
        rates = torch.exp(log_kmer_rates + log_site_rates)
        return rates.squeeze()


# original version
#     def forward(self, encoded_parent):
#         log_kmer_rates = self.kmer_embedding(encoded_parent).squeeze()
#         positions = torch.arange(encoded_parent.size(0), device=encoded_parent.device)
#         log_site_rates = self.log_site_rates(positions).squeeze()
# 
#         # Rates are the product of kmer and site rates.
#         rates = torch.exp(log_kmer_rates + log_site_rates)
#         return rates

    @property
    def kmer_rates(self):
        # Convert kmer log rates to linear space
        return torch.exp(self.kmer_embedding.weight).squeeze()

    @property
    def site_rates(self):
        # Convert site log rates to linear space
        return torch.exp(self.log_site_rates.weight).squeeze()


class SHMoofBurrito:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        model,
        learning_rate=0.01,
        l2_regularization_coeff=1e-6,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=l2_regularization_coeff,
        )

    def train(self, epochs):
        self.model.train()  # Set the model to training mode
        training_losses = []
        validation_losses = []

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
            training_losses.append(epoch_loss)

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
            validation_losses.append(validation_loss)

            print(
                f"Epoch [{epoch+1}/{epochs}]\t Loss: {epoch_loss:.8f}\t Val Loss: {validation_loss:.8f}"
            )

        return pd.DataFrame(
            {"training_losses": training_losses, "validation_losses": validation_losses}
        )

    def _calculate_loss(self, encoded_parent, mask, mutation_indicator):
        rates = self.model(encoded_parent)
        # Note that our mutation frequency has the number of non-N bases in the denominator.
        mutation_freq = (mutation_indicator / mask.sum()).sum()
        mut_prob = 1 - torch.exp(-rates * mutation_freq)
        mut_prob_masked = mut_prob[mask]
        mutation_indicator_masked = mutation_indicator[mask].float()
        return self.criterion(mut_prob_masked, mutation_indicator_masked)

    def write_shmoof_output(self, out_dir):
        # Extract k-mer (motif) mutabilities
        kmer_rates = self.model.kmer_rates.detach().numpy().flatten()
        motif_mutabilities = pd.DataFrame(
            {
                "Motif": self.train_dataset.all_kmers,
                "Mutability": kmer_rates,
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
