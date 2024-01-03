import copy
from abc import ABC, abstractmethod
import os

import pandas as pd
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tensorboardX import SummaryWriter

from netam.common import generate_kmers, kmer_to_index_of, mask_tensor_of
from netam import models


def load_shmoof_dataframes(csv_path, sample_count=None, val_nickname="13"):
    """Load the shmoof dataframes from the csv_path and return train and validation dataframes.

    Args:
        csv_path (str): Path to the csv file containing the shmoof data.
        sample_count (int, optional): Number of samples to use. Defaults to None.
        val_nickname (str, optional): Nickname of the sample to use for validation. Defaults to "13".

    Returns:
        tuple: Tuple of train and validation dataframes.

    Notes:

    The sample nicknames are: `51` is the biggest one, `13` is the second biggest,
    and `small` is the rest of the repertoires merged together.

    If the nickname is `split`, then we do a random 80/20 split of the data.

    Here are the value_counts:
    51       22424
    13       13186
    59        4686
    88        3067
    97        3028
    small     2625
    """
    full_shmoof_df = pd.read_csv(csv_path, index_col=0).reset_index(drop=True)

    # only keep rows where parent is different than child
    full_shmoof_df = full_shmoof_df[full_shmoof_df["parent"] != full_shmoof_df["child"]]

    if sample_count is not None:
        full_shmoof_df = full_shmoof_df.sample(sample_count)

    if val_nickname == "split":
        train_df = full_shmoof_df.sample(frac=0.8)
        val_df = full_shmoof_df.drop(train_df.index)
        return train_df, val_df

    # else
    full_shmoof_df["nickname"] = full_shmoof_df["sample_id"].astype(str).str[-2:]
    for small_nickname in ["80", "37", "50", "07"]:
        full_shmoof_df.loc[
            full_shmoof_df["nickname"] == small_nickname, "nickname"
        ] = "small"

    val_df = full_shmoof_df[full_shmoof_df["nickname"] == val_nickname]
    train_df = full_shmoof_df.drop(val_df.index)

    assert len(val_df) > 0, f"No validation samples found with nickname {val_nickname}"

    return train_df, val_df


def create_mutation_indicator(parent, child, site_count):
    assert len(parent) == len(child), f"{parent} and {child} are not the same length"
    mutation_indicator = [
        1 if parent[i] != child[i] else 0 for i in range(min(len(parent), site_count))
    ]

    # Pad the mutation indicator if necessary
    if len(mutation_indicator) < site_count:
        mutation_indicator += [0] * (site_count - len(mutation_indicator))

    return torch.tensor(mutation_indicator, dtype=torch.bool)


class KmerSequenceEncoder:
    def __init__(self, kmer_length, site_count):
        self.kmer_length = kmer_length
        self.site_count = site_count
        assert kmer_length % 2 == 1
        self.overhang_length = (kmer_length - 1) // 2
        self.all_kmers = generate_kmers(kmer_length)
        self.kmer_to_index = kmer_to_index_of(self.all_kmers)

    @property
    def parameters(self):
        return {"kmer_length": self.kmer_length, "site_count": self.site_count}

    def encode_sequence(self, sequence):
        # Pad sequence with overhang_length 'N's at the start and end so that we
        # can assign parameters to every site in the sequence.
        padded_sequence = (
            "N" * self.overhang_length + sequence + "N" * self.overhang_length
        )

        # Note that we are using a default value of 0 here. So we use the
        # catch-all term for anything with an N in it for the sites on the
        # boundary of the kmer.
        # Note that this line also effectively pads things out to site_count because
        # when i gets large the slice will be empty and we will get a 0.
        # These sites will get masked out by the mask below.
        kmer_indices = [
            self.kmer_to_index.get(padded_sequence[i : i + self.kmer_length], 0)
            for i in range(self.site_count)
        ]

        mask = mask_tensor_of(sequence, self.site_count)

        return torch.tensor(kmer_indices, dtype=torch.int32), mask


class PlaceholderEncoder:
    def __init__(self):
        pass

    @property
    def parameters(self):
        return {}


class SHMoofDataset(Dataset):
    def __init__(self, dataframe, kmer_length, site_count):
        super().__init__()
        self.encoder = KmerSequenceEncoder(kmer_length, site_count)
        (
            self.encoded_parents,
            self.masks,
            self.mutation_indicators,
        ) = self.encode_pcps(dataframe)

    def __len__(self):
        return len(self.encoded_parents)

    def __getitem__(self, idx):
        return self.encoded_parents[idx], self.masks[idx], self.mutation_indicators[idx]

    def __repr__(self):
        return f"{self.__class__.__name__}(Size: {len(self)}) on {self.encoded_parents.device}"

    def to(self, device):
        self.encoded_parents = self.encoded_parents.to(device)
        self.masks = self.masks.to(device)
        self.mutation_indicators = self.mutation_indicators.to(device)

    def encode_pcps(self, dataframe):
        encoded_parents = []
        masks = []
        mutation_vectors = []

        for _, row in dataframe.iterrows():
            encoded_parent, mask = self.encoder.encode_sequence(row["parent"])
            mutation_indicator = create_mutation_indicator(
                row["parent"], row["child"], self.encoder.site_count
            )

            encoded_parents.append(encoded_parent)
            masks.append(mask)
            mutation_vectors.append(mutation_indicator)

        return (
            torch.stack(encoded_parents),
            torch.stack(masks),
            torch.stack(mutation_vectors),
        )


class Crepe:
    """
    A lightweight wrapper around a model that can be used for prediction but not training.
    It handles serialization.
    """

    SERIALIZATION_VERSION = 0

    def __init__(self, encoder, model, training_hyperparameters={}):
        super().__init__()
        self.encoder = encoder
        self.model = model
        self.training_hyperparameters = training_hyperparameters
        self.device = None

    def to(self, device):
        self.device = device
        self.model.to(device)

    def encode_sequences(self, sequences):
        encoded_parents, masks = zip(
            *[self.encoder.encode_sequence(sequence) for sequence in sequences]
        )
        return torch.stack(encoded_parents), torch.stack(masks)

    def __call__(self, sequences):
        encoded_parents, masks = self.encode_sequences(sequences)
        if self.device is not None:
            encoded_parents = encoded_parents.to(self.device)
            masks = masks.to(self.device)
        return self.model(encoded_parents, masks)

    def save(self, prefix):
        torch.save(self.model.state_dict(), f"{prefix}.pth")
        with open(f"{prefix}.yml", "w") as f:
            yaml.dump(
                {
                    "serialization_version": self.SERIALIZATION_VERSION,
                    "model_class": self.model.__class__.__name__,
                    "model_hyperparameters": self.model.hyperparameters,
                    "training_hyperparameters": self.training_hyperparameters,
                    "encoder_class": self.encoder.__class__.__name__,
                    "encoder_parameters": self.encoder.parameters,
                },
                f,
            )


def load_crepe(prefix, device=None):
    with open(f"{prefix}.yml", "r") as f:
        config = yaml.safe_load(f)

    if config["serialization_version"] != Crepe.SERIALIZATION_VERSION:
        raise ValueError(
            f"Unsupported serialization version: {config['serialization_version']}"
        )

    encoder_class_name = config["encoder_class"]

    try:
        encoder_class = globals()[encoder_class_name]
    except AttributeError:
        raise ValueError(f"Encoder class '{encoder_class_name}' not known.")

    encoder = encoder_class(**config["encoder_parameters"])

    model_class_name = config["model_class"]

    try:
        model_class = getattr(models, model_class_name)
    except AttributeError:
        raise ValueError(
            f"Model class '{model_class_name}' not found in 'models' module."
        )

    model = model_class(**config["model_hyperparameters"])

    model_state_path = f"{prefix}.pth"
    if device is None:
        device = torch.device("cpu")
    model.load_state_dict(torch.load(model_state_path, map_location=device))
    model.eval()

    crepe_instance = Crepe(encoder, model, config["training_hyperparameters"])
    if device:
        crepe_instance.to(device)

    return crepe_instance


def crepe_exists(prefix):
    return os.path.exists(f"{prefix}.yml") and os.path.exists(f"{prefix}.pth")


class Burrito(ABC):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        model,
        batch_size=1024,
        learning_rate=0.1,
        min_learning_rate=1e-4,
        l2_regularization_coeff=1e-6,
        verbose=False,
    ):
        """
        Note that we allow train_dataset to be None, to support use cases where
        we just want to do evaluation.
        """
        if train_dataset is None:
            self.train_loader = None
        else:
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.model = model
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.l2_regularization_coeff = l2_regularization_coeff
        self.verbose = verbose
        self.reset_optimization()
        self.writer = SummaryWriter(log_dir="./_logs")
        self.bce_loss = nn.BCELoss()
        self.global_epoch = 0

    def reset_optimization(self):
        """Reset the optimizer and scheduler."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_regularization_coeff,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=4, verbose=self.verbose
        )

    def multi_train(self, epochs, max_tries=3):
        """Train the model. If lr isn't below min_lr, reset the optimizer and scheduler, and reset the model and resume training."""
        for i in range(max_tries):
            train_history = self.train(epochs)
            if self.optimizer.param_groups[0]["lr"] < self.min_learning_rate:
                return train_history
            else:
                print(
                    f"Learning rate {self.optimizer.param_groups[0]['lr']} not below {self.min_learning_rate}. Resetting model and optimizer."
                )
                self.reset_optimization()
                self.model.reinitialize_weights()
        print(f"Failed to train model to min_learning_rate after {max_tries} tries.")
        return train_history

    def process_data_loader(self, data_loader, train_mode=False):
        """
        Process data through the model using the given data loader.
        If train_mode is True, performs optimization steps.

        Args:
            data_loader (DataLoader): DataLoader to use.
            train_mode (bool, optional): Whether to do optimization as part of
                the forward pass. Defaults to False.
                Note that this also applies the regularization loss if set to True.

        Returns:
            float: Average loss.
        """
        total_loss = 0.0
        total_samples = 0

        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(train_mode):
            for batch in data_loader:
                loss = self.loss_of_batch(batch)

                if train_mode:
                    if hasattr(self.model, "regularization_loss"):
                        reg_loss = self.model.regularization_loss()
                        loss += reg_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # We support both dicts and lists of tensors as the batch.
                first_value_of_batch = (
                    list(batch.values())[0] if isinstance(batch, dict) else batch[0]
                )
                batch_size = first_value_of_batch.shape[0]
                # If we multiply the loss by the batch size, then the loss will be the sum of the
                # losses for each example in the batch. Then, when we divide by the number of
                # examples in the dataset below, we will get the average loss per example.
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        average_loss = total_loss / total_samples
        return average_loss

    def train(self, epochs):
        assert self.train_loader is not None, "No training data provided."

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_model_state = None

        def record_losses(train_loss, val_loss):
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.writer.add_scalar("Training loss", train_loss, self.global_epoch)
            self.writer.add_scalar("Validation loss", val_loss, self.global_epoch)

        # Record the initial loss before training.
        train_loss = self.process_data_loader(self.train_loader, train_mode=False)
        val_loss = self.process_data_loader(self.val_loader, train_mode=False)
        record_losses(train_loss, val_loss)

        with tqdm(range(1, epochs + 1), desc="Epoch") as pbar:
            for epoch in pbar:
                current_lr = self.optimizer.param_groups[0]["lr"]
                if current_lr < self.min_learning_rate:
                    break

                train_loss = self.process_data_loader(
                    self.train_loader, train_mode=True
                )
                val_loss = self.process_data_loader(self.val_loader, train_mode=False)
                self.scheduler.step(val_loss)
                record_losses(train_loss, val_loss)
                self.global_epoch += 1

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())

                current_lr = self.optimizer.param_groups[0]["lr"]
                if len(val_losses) > 1:
                    val_loss = val_losses[-1]
                    loss_diff = val_losses[-1] - val_losses[-2]
                    pbar.set_postfix(
                        val_loss=f"{val_loss:.4g}",
                        loss_diff=f"{loss_diff:.4g}",
                        lr=current_lr,
                        refresh=True,
                    )
                self.writer.flush()

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Make sure that saving the best model state worked.
        if (
            best_val_loss != float("inf") # We actually have a training step.
            and abs(best_val_loss - self.evaluate()) > 1e-6
        ):
            print(
                f"\nWarning: observed val loss is {best_val_loss} and saved loss is {self.evaluate()}"
            )

        return pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses})

    def evaluate(self):
        """
        Evaluate the model on the validation set.
        """
        return self.process_data_loader(self.val_loader, train_mode=False)

    def save_crepe(self, prefix):
        self.to_crepe().save(prefix)

    @abstractmethod
    def loss_of_batch(self, batch):
        pass

    @abstractmethod
    def full_train(self, *args, **kwargs):
        pass

    @abstractmethod
    def to_crepe(self):
        pass


class SHMBurrito(Burrito):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        model,
        batch_size=1024,
        learning_rate=0.1,
        min_learning_rate=1e-4,
        l2_regularization_coeff=1e-6,
        verbose=False,
    ):
        super().__init__(
            train_dataset,
            val_dataset,
            model,
            batch_size,
            learning_rate,
            min_learning_rate,
            l2_regularization_coeff,
            verbose,
        )

    def loss_of_batch(self, batch):
        encoded_parents, masks, mutation_indicators = batch
        rates = self.model(encoded_parents, masks)
        mutation_freq = mutation_indicators.sum(dim=1, keepdim=True) / masks.sum(
            dim=1, keepdim=True
        )
        mut_prob = 1 - torch.exp(-rates * mutation_freq)
        mut_prob_masked = mut_prob[masks]
        mutation_indicator_masked = mutation_indicators[masks].float()
        loss = self.bce_loss(mut_prob_masked, mutation_indicator_masked)
        return loss

    def full_train(self, *args, **kwargs):
        return self.train(*args, **kwargs)

    def to_crepe(self):
        training_hyperparameters = {
            key: self.__dict__[key]
            for key in [
                "learning_rate",
                "min_learning_rate",
                "l2_regularization_coeff",
            ]
        }
        encoder = KmerSequenceEncoder(
            self.model.hyperparameters["kmer_length"],
            self.train_loader.dataset.encoder.site_count,
        )
        return Crepe(encoder, self.model, training_hyperparameters)
