import itertools

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tensorboardX import SummaryWriter

from epam.torch_common import PositionalEncoding

BASES = ["A", "C", "G", "T"]


def load_shmoof_dataframes(csv_path, sample_count=None):
    full_shmoof_df = pd.read_csv(csv_path, index_col=0).reset_index(drop=True)

    # only keep rows where parent is different than child
    full_shmoof_df = full_shmoof_df[full_shmoof_df["parent"] != full_shmoof_df["child"]]

    if sample_count is not None:
        full_shmoof_df = full_shmoof_df.sample(sample_count)

    # train_df = full_shmoof_df.sample(frac=0.8)
    # val_df = full_shmoof_df.drop(train_df.index)

    # Hold out sample 326713
    val_df = full_shmoof_df[full_shmoof_df["sample_id"] == 326713]
    train_df = full_shmoof_df.drop(val_df.index)

    return train_df, val_df


def create_mutation_indicator(parent, child, max_length):
    assert len(parent) == len(child), f"{parent} and {child} are not the same length"
    mutation_indicator = [
        1 if parent[i] != child[i] else 0 for i in range(min(len(parent), max_length))
    ]

    # Pad the mutation indicator if necessary
    if len(mutation_indicator) < max_length:
        mutation_indicator += [0] * (max_length - len(mutation_indicator))

    return torch.tensor(mutation_indicator, dtype=torch.bool)


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
            self.mutation_indicators,
        ) = self.encode_sequences(dataframe)

    def __len__(self):
        return len(self.encoded_parents)

    def __getitem__(self, idx):
        return self.encoded_parents[idx], self.masks[idx], self.mutation_indicators[idx]

    def to(self, device):
        self.encoded_parents = self.encoded_parents.to(device)
        self.masks = self.masks.to(device)
        self.mutation_indicators = self.mutation_indicators.to(device)

    def encode_sequences(self, dataframe):
        encoded_parents = []
        masks = []
        mutation_vectors = []

        for _, row in dataframe.iterrows():
            encoded_parent, mask = self.encode_sequence(row["parent"])
            mutation_indicator = create_mutation_indicator(
                row["parent"], row["child"], self.max_length
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


class Burrito:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        model,
        batch_size=1024,
        learning_rate=0.1,
        l2_regularization_coeff=1e-6,
    ):
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.model = model
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=l2_regularization_coeff,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=4, verbose=True
        )

    def train(self, epochs):
        writer = SummaryWriter(log_dir="./_logs")

        self.model.train()
        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            training_loss = 0.0
            for encoded_parents, masks, mutation_indicators in self.train_loader:
                loss = self._calculate_loss(encoded_parents, masks, mutation_indicators)

                if hasattr(self.model, "regularization_loss"):
                    reg_loss = self.model.regularization_loss()
                    loss += reg_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # If we multiply the loss by the batch size, then the loss will be the sum of the
                # losses for each example in the batch. Then, when we divide by the number of
                # examples in the dataset below, we will get the average loss per example.
                training_loss += loss.item() * encoded_parents.size(0)

            training_loss /= len(self.train_loader.dataset)
            training_losses.append(training_loss)

            # Validation phase
            self.model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for encoded_parents, masks, mutation_indicators in self.val_loader:
                    loss = self._calculate_loss(
                        encoded_parents, masks, mutation_indicators
                    )
                    validation_loss += loss.item() * encoded_parents.size(0)
            validation_loss /= len(self.val_loader.dataset)
            validation_losses.append(validation_loss)

            writer.add_scalar("Train loss", training_loss, epoch)
            writer.add_scalar("Validation loss", validation_loss, epoch)

            print(
                f"Epoch [{epoch+1}/{epochs}]\t Loss: {training_loss:.8g}\t Val Loss: {validation_loss:.8g}"
            )
            self.scheduler.step(validation_loss)

        return pd.DataFrame(
            {"training_losses": training_losses, "validation_losses": validation_losses}
        )

    def _calculate_loss(self, encoded_parents, masks, mutation_indicators):
        rates = self.model(encoded_parents, masks)
        mutation_freq = mutation_indicators.sum(dim=1, keepdim=True) / masks.sum(
            dim=1, keepdim=True
        )
        mut_prob = 1 - torch.exp(-rates * mutation_freq)
        mut_prob_masked = mut_prob[masks]
        mutation_indicator_masked = mutation_indicators[masks].float()
        loss = self.criterion(mut_prob_masked, mutation_indicator_masked)
        return loss