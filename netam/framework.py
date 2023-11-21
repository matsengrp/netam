from datetime import datetime
import inspect
import itertools

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna

from tensorboardX import SummaryWriter

from epam.torch_common import PositionalEncoding

BASES = ["A", "C", "G", "T"]

bce_loss = nn.BCELoss()

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
    51       38174
    small    24208
    13       21940

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
    full_shmoof_df["nickname"] = "small"
    full_shmoof_df.loc[full_shmoof_df["sample_id"] == 326651, "nickname"] = "51"
    full_shmoof_df.loc[full_shmoof_df["sample_id"] == 326713, "nickname"] = "13"

    val_df = full_shmoof_df[full_shmoof_df["nickname"] == val_nickname]
    train_df = full_shmoof_df.drop(val_df.index)

    assert len(val_df) > 0, f"No validation samples found with nickname {val_nickname}"

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


def filter_kwargs(func, kwargs):
    """
    Filter kwargs to only those that the function accepts.
    """
    # Get the parameters of the function
    sig = inspect.signature(func)
    func_params = sig.parameters

    # Filter kwargs to only those that the function accepts
    return {k: v for k, v in kwargs.items() if k in func_params}


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


def calculate_loss(rates, masks, mutation_indicators):
    mutation_freq = mutation_indicators.sum(dim=1, keepdim=True) / masks.sum(
        dim=1, keepdim=True
    )
    mut_prob = 1 - torch.exp(-rates * mutation_freq)
    mut_prob_masked = mut_prob[masks]
    mutation_indicator_masked = mutation_indicators[masks].float()
    loss = bce_loss(mut_prob_masked, mutation_indicator_masked)
    return loss


class Burrito:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        model,
        batch_size=1024,
        learning_rate=0.1,
        min_learning_rate=1e-4,
        l2_regularization_coeff=1e-6,
        verbose=True,
    ):
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=l2_regularization_coeff,
        )
        self.min_learning_rate = min_learning_rate
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=4, verbose=verbose
        )
        self.verbose = verbose

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
            for encoded_parents, masks, mutation_indicators in data_loader:
                rates = self.model(encoded_parents, masks)
                loss = calculate_loss(rates, masks, mutation_indicators)

                if train_mode:
                    if hasattr(self.model, "regularization_loss"):
                        reg_loss = self.model.regularization_loss()
                        loss += reg_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # If we multiply the loss by the batch size, then the loss will be the sum of the
                # losses for each example in the batch. Then, when we divide by the number of
                # examples in the dataset below, we will get the average loss per example.
                total_loss += loss.item() * encoded_parents.size(0)
                total_samples += encoded_parents.size(0)

        average_loss = total_loss / total_samples
        return average_loss


    def train(self, epochs):
        writer = SummaryWriter(log_dir="./_logs")
        train_losses = []
        val_losses = []

        def record_losses(epoch, train_loss, val_loss):
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            writer.add_scalar("Train loss", train_loss, epoch)
            writer.add_scalar("Validation loss", val_loss, epoch)

            if self.verbose:
                print(
                    f"Epoch [{epoch}/{epochs}]\t Loss: {train_loss:.8g}\t Val Loss: {val_loss:.8g}"
                )

        train_loss = self.process_data_loader(self.train_loader, train_mode=False)
        val_loss = self.process_data_loader(self.val_loader, train_mode=False)

        record_losses(0, train_loss, val_loss)

        for epoch in range(1, epochs+1):
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr < self.min_learning_rate:
                if self.verbose:
                    print(f"Stopping training early: learning rate below {self.min_learning_rate}")
                break

            train_loss = self.process_data_loader(self.train_loader, train_mode=True)
            val_loss = self.process_data_loader(self.val_loader, train_mode=False)
            record_losses(epoch, train_loss, val_loss)

            self.scheduler.step(val_loss)

        return pd.DataFrame(
            {"train_loss": train_losses, "val_loss": val_losses}
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


class HyperBurrito:
    """
    A burrito that can be used to optimize hyperparameters.
    """
    def __init__(
        self,
        device,
        train_dataset,
        val_dataset,
        model_class,
        epochs=100,
    ):
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        train_dataset.to(self.device)
        val_dataset.to(self.device)
        self.model_class = model_class
        self.epochs = epochs
        
    def burrito_of_model(self, model, **kwargs):
        burrito = Burrito(
            self.train_dataset,
            self.val_dataset,
            model,
            verbose=False,
            **kwargs
        )
        return burrito

    def optuna_objective(self, trial, int_params, cat_params, float_params, log_float_params, fixed_hyperparams=None):
        """ Optuna objective function.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            int_params (dict): Dictionary of integer parameters to optimize.
            cat_params (dict): Dictionary of categorical parameters to optimize.
            float_params (dict): Dictionary of float parameters to optimize on a linear scale.
            log_float_params (dict): Dictionary of float parameters to optimize on a log scale.
            fixed_hyperparams (dict, optional): Dictionary of fixed hyperparameters. Defaults to None.
            
        Returns:
            float: Validation loss.
        """
        hyperparams = fixed_hyperparams or {}

        for param_name, choices in cat_params.items():
            hyperparams[param_name] = trial.suggest_categorical(param_name, choices)

        for param_name, (low, high) in int_params.items():
            hyperparams[param_name] = trial.suggest_int(param_name, low, high)

        for param_name, (low, high) in float_params.items():
            hyperparams[param_name] = trial.suggest_float(param_name, low, high)

        for param_name, (low, high) in log_float_params.items():
            hyperparams[param_name] = trial.suggest_float(param_name, low, high, log=True)

        model_hyperparams = filter_kwargs(self.model_class, hyperparams)
        model = self.model_class(self.train_dataset, **model_hyperparams)
        model.to(self.device)

        burrito_hyperparams = filter_kwargs(self.burrito_of_model, hyperparams)
        burrito = self.burrito_of_model(model, **burrito_hyperparams)

        losses = burrito.train(epochs=self.epochs)

        return losses["validation_losses"].min()


    def optuna_optimize(self, n_trials, cat_params, int_params, float_params, log_float_params, fixed_hyperparams=None):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.optuna_objective(trial, int_params, cat_params, float_params, log_float_params, fixed_hyperparams), n_trials=n_trials)
        best_hyperparams = study.best_params
        best_score = study.best_value
        print(f"Best Hyperparameters: {best_hyperparams}")
        print(f"Best Validation Loss: {best_score}")

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"_ignore/optuna_{self.model_class.__name__}_{timestamp_str}.csv"
        trial_data = study.trials_dataframe()
        trial_data.to_csv(output_path, index=False)
 