"""
A silly amino acid prediction model.

We'll use these conventions:

* B is the batch size
* L is the max sequence length

"""

import math
import os

import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

from epam.torch_common import pick_device, PositionalEncoding
import epam.sequences as sequences


class AAPCPDataset(Dataset):
    def __init__(self, aa_parents, aa_children):
        assert len(aa_parents) == len(aa_parents)
        pcp_count = len(aa_parents)

        for parent, child in zip(aa_parents, aa_children):
            if parent == child:
                raise ValueError(
                    f"Found an identical parent and child sequence: {parent}"
                )

        self.max_aa_seq_len = max(len(seq) for seq in aa_parents)
        self.aa_parents_onehot = torch.zeros((pcp_count, self.max_aa_seq_len, 20))
        self.aa_subs_indicator_tensor = torch.zeros((pcp_count, self.max_aa_seq_len))

        # padding_mask is True for padding positions.
        self.padding_mask = torch.ones(
            (pcp_count, self.max_aa_seq_len), dtype=torch.bool
        )

        for i, (aa_parent, aa_child) in enumerate(zip(aa_parents, aa_children)):
            aa_indices_parent = sequences.aa_idx_array_of_str(aa_parent)
            aa_seq_len = len(aa_parent)
            self.aa_parents_onehot[i, torch.arange(aa_seq_len), aa_indices_parent] = 1
            self.aa_subs_indicator_tensor[i, :aa_seq_len] = torch.tensor(
                [p != c for p, c in zip(aa_parent, aa_child)], dtype=torch.float
            )
            self.padding_mask[i, :aa_seq_len] = False

    def __len__(self):
        return len(self.aa_parents_onehot)

    def __getitem__(self, idx):
        return {
            "aa_onehot": self.aa_parents_onehot[idx],
            "subs_indicator": self.aa_subs_indicator_tensor[idx],
            "padding_mask": self.padding_mask[idx],
        }


class TransformerBinaryModel(nn.Module):
    """A transformer-based model for binary selection.

    This is a model that takes in a batch of one-hot encoded sequences and outputs a binary selection matrix.

    See forward() for details.
    """

    def __init__(
        self,
        nhead: int,
        dim_feedforward: int,
        layer_count: int,
        d_model: int = 20,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.device = pick_device()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layer_count)
        self.linear = nn.Linear(self.d_model, 1)

        self.to(self.device)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, parent_onehots: Tensor, padding_mask: Tensor) -> Tensor:
        """Build a binary log selection matrix from a one-hot encoded parent sequence.

        Because we're predicting log of the selection factor, we don't use an
        activation function after the transformer.

        Parameters:
            parent_onehots: A tensor of shape (B, L, 20) representing the one-hot encoding of parent sequences.
            padding_mask: A tensor of shape (B, L) representing the padding mask for the sequence.

        Returns:
            A tensor of shape (B, L, 1) representing the log level of selection
            for each amino acid site.
        """

        parent_onehots = parent_onehots * math.sqrt(self.d_model)
        # Have to do the permutation because the positional encoding expects the
        # sequence length to be the first dimension.
        parent_onehots = self.pos_encoder(parent_onehots.permute(1, 0, 2)).permute(
            1, 0, 2
        )

        # NOTE: not masking due to MPS bug
        out = self.encoder(parent_onehots)  # , src_key_padding_mask=padding_mask)
        out = self.linear(out)
        return out.squeeze(-1)

    def prediction_of_aa_str(self, aa_str: str):
        """Do the forward method without gradients from an amino acid string and convert to numpy.

        Parameters:
            aa_str: A string of amino acids.

        Returns:
            A numpy array of the same length as the input string representing
            the level of selection for each amino acid site.
        """
        aa_onehot = sequences.aa_onehot_tensor_of_str(aa_str)

        # Create a padding mask with False values (i.e., no padding)
        padding_mask = torch.zeros(len(aa_str), dtype=torch.bool).to(self.device)

        with torch.no_grad():
            aa_onehot = aa_onehot.to(self.device)
            model_out = self(aa_onehot.unsqueeze(0), padding_mask.unsqueeze(0)).squeeze(
                0
            )
            final_out = torch.exp(model_out)
            final_out = torch.clamp(final_out, min=0.0, max=0.999)

        return final_out.cpu().numpy()


def train_model(
    pcp_df,
    nhead,
    dim_feedforward,
    layer_count,
    batch_size=32,
    num_epochs=10,
    learning_rate=0.001,
    checkpoint_dir="./_checkpoints",
    log_dir="./_logs",
):
    print("preparing data...")
    parents = pcp_df["aa_parent"]
    children = pcp_df["aa_child"]

    train_len = int(0.8 * len(parents))
    train_parents, val_parents = parents[:train_len], parents[train_len:]
    train_children, val_children = children[:train_len], children[train_len:]

    # It's important to make separate PCPDatasets for training and validation
    # because the maximum sequence length can differ between those two.
    train_set = AAPCPDataset(train_parents, train_children)
    val_set = AAPCPDataset(val_parents, val_children)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = TransformerBinaryModel(
        nhead=nhead, dim_feedforward=dim_feedforward, layer_count=layer_count
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = model.device
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    bce_loss = nn.BCELoss()

    def complete_loss_fn(log_aa_mut_probs, aa_subs_indicator, padding_mask):
        predictions = torch.exp(log_aa_mut_probs)

        predictions = predictions.masked_select(~padding_mask)
        aa_subs_indicator = aa_subs_indicator.masked_select(~padding_mask)

        # In the early stages of training, we can get probabilities > 1.0 because
        # of bad parameter initialization. We clamp the predictions to be between
        # 0 and 0.999 to avoid this: out of range predictions can make NaNs
        # downstream.
        out_of_range_prediction_count = torch.sum(predictions > 1.0)
        # if out_of_range_prediction_count > 0:
        #     print(f"{out_of_range_prediction_count}\tpredictions out of range.")
        predictions = torch.clamp(predictions, min=0.0, max=0.999)

        return bce_loss(predictions, aa_subs_indicator)

    def loss_of_batch(batch):
        aa_onehot = batch["aa_onehot"].to(device)
        aa_subs_indicator = batch["subs_indicator"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        log_aa_mut_probs = model(aa_onehot, padding_mask)
        return complete_loss_fn(
            log_aa_mut_probs,
            aa_subs_indicator,
            padding_mask,
        )

    def compute_avg_loss(data_loader):
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                total_loss += loss_of_batch(batch).item()
        return total_loss / len(data_loader)

    # Record epoch 0
    model.eval()
    avg_train_loss_epoch_zero = compute_avg_loss(train_loader)
    avg_val_loss_epoch_zero = compute_avg_loss(val_loader)
    writer.add_scalar("Training Loss", avg_train_loss_epoch_zero, 0)
    writer.add_scalar("Validation Loss", avg_val_loss_epoch_zero, 0)
    print(
        f"Epoch [0/{num_epochs}], Training Loss: {avg_train_loss_epoch_zero}, Validation Loss: {avg_val_loss_epoch_zero}"
    )

    loss_records = []

    for epoch in range(num_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = loss_of_batch(batch)
            loss.backward()
            optimizer.step()
            writer.add_scalar(
                "Training Loss", loss.item(), epoch * len(train_loader) + i
            )

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += loss_of_batch(batch).item()

            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar("Validation Loss", avg_val_loss, epoch)

            # Save model checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_val_loss,
                },
                f"{checkpoint_dir}/model_epoch_{epoch}.pth",
            )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}"
        )

        loss_records.append(
            {
                "Epoch": epoch + 1,
                "Training Loss": loss.item(),
                "Validation Loss": avg_val_loss,
            }
        )

    writer.close()
    loss_df = pd.DataFrame(loss_records)
    loss_df.to_csv("training_validation_loss.csv", index=False)

    return model
