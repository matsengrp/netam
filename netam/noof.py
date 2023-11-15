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


class NoofBurrito:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        model,
        batch_size=1024,
        learning_rate=0.01,
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

    def train(self, epochs):
        writer = SummaryWriter(log_dir="./_logs")

        self.model.train()
        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            training_loss = 0.0
            for encoded_parents, masks, mutation_indicators in self.train_loader:
                loss = self._calculate_loss(encoded_parents, masks, mutation_indicators)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                training_loss += loss.item()

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
                    validation_loss += loss.item()
            validation_loss /= len(self.val_loader.dataset)
            validation_losses.append(validation_loss)

            writer.add_scalar("Train loss", training_loss, epoch)
            writer.add_scalar("Validation loss", validation_loss, epoch)

            print(
                f"Epoch [{epoch+1}/{epochs}]\t Loss: {training_loss:.8g}\t Val Loss: {validation_loss:.8g}"
            )

        return pd.DataFrame(
            {"training_losses": training_losses, "validation_losses": validation_losses}
        )

    def _calculate_loss(self, encoded_parents, masks, mutation_indicators):
        rates = self.model(encoded_parents)
        mutation_freq = mutation_indicators.sum(dim=1, keepdim=True) / masks.sum(dim=1, keepdim=True)
        mut_prob = 1 - torch.exp(-rates * mutation_freq)
        mut_prob_masked = mut_prob[masks]
        mutation_indicator_masked = mutation_indicators[masks].float()
        loss = self.criterion(mut_prob_masked, mutation_indicator_masked)
        return loss
