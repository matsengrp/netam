import copy
from abc import ABC, abstractmethod
import os

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tensorboardX import SummaryWriter

from netam.common import (
    generate_kmers,
    kmer_to_index_of,
    nt_mask_tensor_of,
    BASES,
    BASES_AND_N_TO_INDEX,
    BIG,
    VRC01_NT_SEQ,
)
from netam import models

from epam.torch_common import optimize_branch_length


def encode_mut_pos_and_base(parent, child, site_count=None):
    """
    This function takes a parent and child sequence and returns a tuple of
    tensors: (mutation_indicator, new_base_idxs).
    The mutation_indicator tensor is a boolean tensor indicating whether
    each site is mutated. Both the parent and the child must be one of
    A, C, G, T, to be considered a mutation.
    The new_base_idxs tensor is an integer tensor that gives the index of the
    new base at each site.

    Note that we use -1 as a placeholder for non-mutated bases in the
    new_base_idxs tensor. This ensures that lack of masking will lead
    to an error.

    If site_count is not None, then the tensors will be trimmed & padded to the
    provided length.
    """
    assert len(parent) == len(child), f"{parent} and {child} are not the same length"

    if site_count is None:
        site_count = len(parent)

    mutation_indicator = []
    new_base_idxs = []

    for i in range(min(len(parent), site_count)):
        if parent[i] != child[i] and parent[i] in BASES and child[i] in BASES:
            mutation_indicator.append(1)
            new_base_idxs.append(BASES_AND_N_TO_INDEX[child[i]])
        else:
            mutation_indicator.append(0)
            new_base_idxs.append(-1)  # No mutation, so set to -1

    # Pad the lists if necessary
    if len(mutation_indicator) < site_count:
        padding_length = site_count - len(mutation_indicator)
        mutation_indicator += [0] * padding_length
        new_base_idxs += [-1] * padding_length

    return (
        torch.tensor(mutation_indicator, dtype=torch.bool),
        torch.tensor(new_base_idxs, dtype=torch.int64),
    )


def wt_base_modifier_of(parent, site_count):
    """
    The wt_base_modifier tensor is all 0s except for the wt base at each site,
    which is -BIG.

    We will add wt_base_modifier to the CSP logits. This will zero out the
    prediction of WT at each site after softmax.
    """
    wt_base_modifier = torch.zeros((site_count, 4))
    for i, base in enumerate(parent[:site_count]):
        if base in BASES:
            wt_base_modifier[i, BASES_AND_N_TO_INDEX[base]] = -BIG
    return wt_base_modifier


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
        sequence = sequence.upper()
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

        wt_base_modifier = wt_base_modifier_of(sequence, self.site_count)

        return torch.tensor(kmer_indices, dtype=torch.int32), wt_base_modifier


class PlaceholderEncoder:
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
            self.new_base_idxs,
            self.wt_base_modifier,
            self.branch_lengths,
        ) = self.encode_pcps(dataframe)
        assert self.encoded_parents.shape[0] == self.branch_lengths.shape[0]

    def __len__(self):
        return len(self.encoded_parents)

    def __getitem__(self, idx):
        return (
            self.encoded_parents[idx],
            self.masks[idx],
            self.mutation_indicators[idx],
            self.new_base_idxs[idx],
            self.wt_base_modifier[idx],
            self.branch_lengths[idx],
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(Size: {len(self)}) on {self.encoded_parents.device}"

    def to(self, device):
        self.encoded_parents = self.encoded_parents.to(device)
        self.masks = self.masks.to(device)
        self.mutation_indicators = self.mutation_indicators.to(device)
        self.new_base_idxs = self.new_base_idxs.to(device)
        self.wt_base_modifier = self.wt_base_modifier.to(device)
        self.branch_lengths = self.branch_lengths.to(device)

    def encode_pcps(self, dataframe):
        encoded_parents = []
        masks = []
        mutation_vectors = []
        new_base_idx_vectors = []
        wt_base_modifier_vectors = []
        branch_lengths = []

        for _, row in dataframe.iterrows():
            encoded_parent, wt_base_modifier = self.encoder.encode_sequence(
                row["parent"]
            )
            mask = nt_mask_tensor_of(row["child"], self.encoder.site_count)
            # Assert that anything that is masked in the child is also masked in
            # the parent. We only use the parent_mask for this check.
            parent_mask = nt_mask_tensor_of(row["parent"], self.encoder.site_count)
            assert (mask <= parent_mask).all()
            (
                mutation_indicator,
                new_base_idxs,
            ) = encode_mut_pos_and_base(
                row["parent"], row["child"], self.encoder.site_count
            )

            encoded_parents.append(encoded_parent)
            masks.append(mask)
            mutation_vectors.append(mutation_indicator)
            new_base_idx_vectors.append(new_base_idxs)
            wt_base_modifier_vectors.append(wt_base_modifier)
            # The initial branch lengths are the normalized number of mutations.
            branch_lengths.append(mutation_indicator.sum() / mask.sum())

        return (
            torch.stack(encoded_parents),
            torch.stack(masks),
            torch.stack(mutation_vectors),
            torch.stack(new_base_idx_vectors),
            torch.stack(wt_base_modifier_vectors),
            torch.tensor(branch_lengths),
        )

    def normalized_mutation_frequency(self):
        return self.mutation_indicators.sum(axis=1) / self.masks.sum(axis=1)

    def export_branch_lengths(self, out_csv_path):
        pd.DataFrame(
            {
                "branch_length": self.branch_lengths,
                "mut_freq": self.normalized_mutation_frequency(),
            }
        ).to_csv(out_csv_path, index=False)

    def load_branch_lengths(self, in_csv_path):
        self.branch_lengths = pd.read_csv(in_csv_path)["branch_length"].values


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
        encoded_parents, wt_base_modifiers = zip(
            *[self.encoder.encode_sequence(sequence) for sequence in sequences]
        )
        masks = [nt_mask_tensor_of(sequence, self.encoder.site_count) for sequence in sequences]
        return (
            torch.stack(encoded_parents),
            torch.stack(masks),
            torch.stack(wt_base_modifiers),
        )

    def __call__(self, sequences):
        encoded_parents, masks, wt_base_modifiers = self.encode_sequences(sequences)
        if self.device is not None:
            encoded_parents = encoded_parents.to(self.device)
            masks = masks.to(self.device)
            wt_base_modifiers = wt_base_modifiers.to(self.device)
        with torch.no_grad():
            outputs = self.model(encoded_parents, masks, wt_base_modifiers)
            return tuple(t.detach().cpu() for t in outputs)

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
    assert crepe_exists(prefix), f"Crepe {prefix} not found."
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


def trimmed_shm_model_outputs_of_crepe(crepe, parents):
    """
    Model outputs trimmed to the length of the parent sequences.
    """
    rates, csp_logits = crepe(parents)
    rates = rates.cpu().detach()
    csps = torch.softmax(csp_logits, dim=-1).cpu().detach()
    trimmed_rates = [rates[i, : len(parent)] for i, parent in enumerate(parents)]
    trimmed_csps = [csps[i, : len(parent)] for i, parent in enumerate(parents)]
    return trimmed_rates, trimmed_csps


def load_and_add_shm_model_outputs_to_pcp_df(pcp_df_path_gz, crepe_prefix, device=None):
    pcp_df = pd.read_csv(pcp_df_path_gz, compression="gzip", index_col=0).reset_index(
        drop=True
    )
    crepe = load_crepe(crepe_prefix, device)
    rates, csps = trimmed_shm_model_outputs_of_crepe(crepe, pcp_df["parent"])
    pcp_df["rates"] = rates
    pcp_df["subs_probs"] = csps
    return pcp_df


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
        name="",
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
            self.writer = SummaryWriter(log_dir=f"./_logs/{name}")
            self.writer.add_text("model_name", model.__class__.__name__)
            self.writer.add_text("model_hyperparameters", str(model.hyperparameters))
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.model = model
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.l2_regularization_coeff = l2_regularization_coeff
        self.verbose = verbose
        self.name = name
        self.reset_optimization()
        self.bce_loss = nn.BCELoss()
        self.global_epoch = 0

    @property
    def device(self):
        return next(self.model.parameters()).device

    def reset_optimization(self):
        """Reset the optimizer and scheduler."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_regularization_coeff,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, verbose=self.verbose
        )

    def multi_train(self, epochs, max_tries=3):
        """
        Train the model. If lr isn't below min_lr, reset the optimizer and
        scheduler, and reset the model and resume training.
        """
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

    def write_loss(self, loss_name, loss, step):
        self.writer.add_scalar(loss_name, loss, step)

    def process_data_loader(self, data_loader, train_mode=False, loss_reduction=None):
        """
        Process data through the model using the given data loader.
        If train_mode is True, performs optimization steps.

        Args:
            data_loader (DataLoader): DataLoader to use.
            train_mode (bool, optional): Whether to do optimization as part of
                the forward pass. Defaults to False.
                Note that this also applies the regularization loss if set to True.
            loss_reduction (callable, optional): Function to reduce the loss
                tensor to a scalar. If None, uses torch.sum. Defaults to None.

        Returns:
            float: Average loss.
        """
        total_loss = None
        total_samples = 0

        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        if loss_reduction is None:
            loss_reduction = torch.sum

        with torch.set_grad_enabled(train_mode):
            for batch in data_loader:
                loss = self.loss_of_batch(batch)
                if total_loss is None:
                    total_loss = torch.zeros_like(loss)

                if train_mode:
                    max_grad_retries = 5
                    for grad_retry_count in range(max_grad_retries):
                        scalar_loss = loss_reduction(loss)
                        if hasattr(self.model, "regularization_loss"):
                            reg_loss = self.model.regularization_loss()
                            scalar_loss += reg_loss

                        self.optimizer.zero_grad()
                        scalar_loss.backward()

                        nan_in_gradients = False
                        for name, param in self.model.named_parameters():
                            if torch.isnan(param).any():
                                raise ValueError(f"NaN in weights: {name}")
                            if param.grad is not None and torch.isnan(param.grad).any():
                                nan_in_gradients = True

                        if not nan_in_gradients:
                            self.optimizer.step()
                            break
                        else:
                            if grad_retry_count < max_grad_retries - 1:
                                print(
                                    f"Retrying gradient calculation ({grad_retry_count + 1}/{max_grad_retries}) with loss {torch.sum(loss).item()}"
                                )
                                loss = self.loss_of_batch(batch)
                            else:
                                raise ValueError(f"Exceeded maximum gradient retries!")

                # We support both dicts and lists of tensors as the batch.
                first_value_of_batch = (
                    list(batch.values())[0] if isinstance(batch, dict) else batch[0]
                )
                batch_size = first_value_of_batch.shape[0]
                # If we multiply the loss by the batch size, then the loss will be the sum of the
                # losses for each example in the batch. Then, when we divide by the number of
                # examples in the dataset below, we will get the average loss per example.
                total_loss += loss.detach() * batch_size
                total_samples += batch_size

        average_loss = total_loss / total_samples
        if hasattr(self, "writer"):
            if train_mode:
                self.write_loss("Training loss", average_loss, self.global_epoch)
            else:
                self.write_loss("Validation loss", average_loss, self.global_epoch)
        return loss_reduction(average_loss)

    def train(self, epochs):
        assert self.train_loader is not None, "No training data provided."

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_model_state = None

        def record_losses(train_loss, val_loss):
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Record the initial loss before training.
        train_loss = self.process_data_loader(
            self.train_loader, train_mode=False
        ).item()
        val_loss = self.process_data_loader(self.val_loader, train_mode=False).item()
        record_losses(train_loss, val_loss)

        with tqdm(range(1, epochs + 1), desc="Epoch") as pbar:
            for epoch in pbar:
                current_lr = self.optimizer.param_groups[0]["lr"]
                if current_lr < self.min_learning_rate:
                    break

                train_loss = self.process_data_loader(
                    self.train_loader, train_mode=True
                ).item()
                val_loss = self.process_data_loader(
                    self.val_loader, train_mode=False
                ).item()
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

        return pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses})

    def evaluate(self):
        """
        Evaluate the model on the validation set.
        """
        return self.process_data_loader(self.val_loader, train_mode=False).item()

    def save_crepe(self, prefix):
        self.to_crepe().save(prefix)

    def standardize_model_rates(self):
        """This is an opportunity to standardize the model rates. Only the
        SHMBurrito class implements this, which makes sense because it 
        needs to get normalized but the DNSM does not."""
        pass

    def standardize_and_optimize_branch_lengths(self, **optimization_kwargs):
        self.standardize_model_rates()
        print(f"vrc01 rate {self.vrc01_site_1_model_rate()}")
        if "learning_rate" not in optimization_kwargs:
            optimization_kwargs["learning_rate"] = 0.01
        if "optimization_tol" not in optimization_kwargs:
            optimization_kwargs["optimization_tol"] = 1e-3
        # We do the branch length optimization on CPU but want to restore the
        # model to the device it was on before.
        device = next(self.model.parameters()).device
        self.model.to("cpu")
        for loader in [self.train_loader, self.val_loader]:
            if loader is None:
                continue
            dataset = loader.dataset
            dataset.to("cpu")
            dataset.branch_lengths = self.find_optimal_branch_lengths(
                dataset, **optimization_kwargs
            )
            dataset.to(device)
        self.model.to(device)

    def standardize_and_use_yun_approx_branch_lengths(self):
        """
        Yun Song's approximation to the branch lengths.
        
        This approximation is the mutation count divided by the total mutation rate for the sequence.
        See https://github.com/matsengrp/netam/assets/112708/034abb74-5635-48dc-bf28-4321b9110222
        """
        self.standardize_model_rates()
        for loader in [self.train_loader, self.val_loader]:
            if loader is None:
                continue
            dataset = loader.dataset
            lengths = []
            for (
                encoded_parent,
                mask,
                mutation_indicator,
                wt_base_modifier,
            ) in zip(
                dataset.encoded_parents,
                dataset.masks,
                dataset.mutation_indicators,
                dataset.wt_base_modifier,
            ):
                rates, _ = self.model(
                    encoded_parent.unsqueeze(0),
                    mask.unsqueeze(0),
                    wt_base_modifier.unsqueeze(0),
                )
                mutation_indicator = mutation_indicator[mask].float()
                length = torch.sum(mutation_indicator) / torch.sum(rates) 
                lengths.append(length.item())
            dataset.branch_lengths = torch.tensor(lengths)

    def mark_branch_lengths_optimized(self, cycle):
        self.writer.add_scalar("branch length optimization", cycle, self.global_epoch)

    def joint_train(self, epochs=100, cycle_count=2, training_method="full"):
        """
        Do joint optimization of model and branch lengths.
        
        If training_method is "full", then we optimize the branch lengths using full ML optimization.
        If training_method is "yun", then we use Yun's approximation to the branch lengths.
        If training_method is "fixed", then we fix the branch lengths and only optimize the model.
        """
        if training_method == "full":
            optimize_branch_lengths = self.standardize_and_optimize_branch_lengths
        elif training_method == "yun":
            optimize_branch_lengths = self.standardize_and_use_yun_approx_branch_lengths
        elif training_method == "fixed":
            optimize_branch_lengths = lambda: None
        else:
            raise ValueError(f"Unknown training method {training_method}")
        loss_history_l = []
        self.mark_branch_lengths_optimized(0)
        loss_history_l.append(self.train(3))
        optimize_branch_lengths()
        self.mark_branch_lengths_optimized(0)
        for cycle in range(cycle_count):
            self.mark_branch_lengths_optimized(cycle + 1)
            self.reset_optimization()
            loss_history_l.append(self.train(epochs))
            if cycle < cycle_count - 1:
                optimize_branch_lengths()
            self.mark_branch_lengths_optimized(cycle + 1)

        return pd.concat(loss_history_l, ignore_index=True)

    @abstractmethod
    def loss_of_batch(self, batch):
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
        name="",
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
            name,
        )

    def loss_of_batch(self, batch):
        (
            encoded_parents,
            masks,
            mutation_indicators,
            _,
            wt_base_modifier,
            branch_lengths,
        ) = batch
        rates = self.model(encoded_parents, masks, wt_base_modifier)
        mut_prob = 1 - torch.exp(-rates * branch_lengths.unsqueeze(-1))
        mut_prob_masked = mut_prob[masks]
        mutation_indicator_masked = mutation_indicators[masks].float()
        loss = self.bce_loss(mut_prob_masked, mutation_indicator_masked)
        return loss

    def vrc01_site_1_model_rate(self):
        """
        Calculate rate on site 1 (zero-indexed) of VRC01_NT_SEQ.
        """
        encoder = self.val_loader.dataset.encoder
        assert encoder.site_count >= 2
        encoded_parent, wt_base_modifier = encoder.encode_sequence(VRC01_NT_SEQ)
        mask = nt_mask_tensor_of(VRC01_NT_SEQ, encoder.site_count)
        vrc01_rates, _ = self.model(encoded_parent.unsqueeze(0), mask.unsqueeze(0), wt_base_modifier.unsqueeze(0))
        vrc01_rate_1 = vrc01_rates.squeeze()[1].item()
        return vrc01_rate_1

    def standardize_model_rates(self):
        """
        Normalize the rates output by the model so that it predicts rate 1 on site 1 
        (zero-indexed) of VRC01_NT_SEQ.
        """
        vrc01_rate_1 = self.vrc01_site_1_model_rate()
        self.model.adjust_rate_bias_by(-np.log(vrc01_rate_1))

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


class RSSHMBurrito(SHMBurrito):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xent_loss = nn.CrossEntropyLoss()
        self.loss_weights = torch.tensor([1.0, 0.01]).to(self.device)

    def process_data_loader(self, data_loader, train_mode=False, loss_reduction=None):
        if loss_reduction is None:
            loss_reduction = lambda x: torch.sum(x * self.loss_weights)

        return super().process_data_loader(data_loader, train_mode, loss_reduction)

    def evaluate(self):
        return super().process_data_loader(
            self.val_loader, train_mode=False, loss_reduction=lambda x: x
        )

    def loss_of_batch(self, batch):
        (
            encoded_parents,
            masks,
            mutation_indicators,
            new_base_idxs,
            wt_base_modifier,
            branch_lengths,
        ) = batch
        rates, csp_logits = self.model(encoded_parents, masks, wt_base_modifier)

        mut_prob = 1 - torch.exp(-rates * branch_lengths.unsqueeze(-1))
        mut_prob_masked = mut_prob[masks]
        mutation_indicator_masked = mutation_indicators[masks].float()
        rate_loss = self.bce_loss(mut_prob_masked, mutation_indicator_masked)

        # Conditional substitution probability (CSP) loss calculation
        # Mask the new_base_idxs to focus only on positions with mutations
        mutated_positions_mask = mutation_indicators == 1
        csp_logits_masked = csp_logits[mutated_positions_mask]
        new_base_idxs_masked = new_base_idxs[mutated_positions_mask]
        # Recall that WT bases are encoded as -1 in new_base_idxs_masked, so
        # this assert makes sure that the loss is masked out for WT bases.
        assert (new_base_idxs_masked >= 0).all()
        csp_loss = self.xent_loss(csp_logits_masked, new_base_idxs_masked)

        return torch.stack([rate_loss, csp_loss])

    def _find_optimal_branch_length(
        self,
        encoded_parent,
        mask,
        mutation_indicator,
        wt_base_modifier,
        starting_branch_length,
        **optimization_kwargs,
    ):
        if torch.sum(mutation_indicator) == 0:
            return 0.0

        rates, _ = self.model(
            encoded_parent.unsqueeze(0),
            mask.unsqueeze(0),
            wt_base_modifier.unsqueeze(0),
        )

        rates = rates.squeeze().double()
        mutation_indicator_masked = mutation_indicator[mask].double()

        def log_pcp_probability(log_branch_length):
            branch_length = torch.exp(log_branch_length)
            mut_prob = 1 - torch.exp(-rates * branch_length)
            mut_prob_masked = mut_prob[mask]
            rate_loss = self.bce_loss(mut_prob_masked, mutation_indicator_masked)
            return -rate_loss

        return optimize_branch_length(
            log_pcp_probability,
            starting_branch_length.double().item(),
            **optimization_kwargs,
        )

    def find_optimal_branch_lengths(self, dataset, **optimization_kwargs):
        optimal_lengths = []

        self.model.eval()
        self.model.freeze()

        for (
            encoded_parent,
            mask,
            mutation_indicator,
            wt_base_modifier,
            starting_branch_length,
        ) in tqdm(
            zip(
                dataset.encoded_parents,
                dataset.masks,
                dataset.mutation_indicators,
                dataset.wt_base_modifier,
                dataset.branch_lengths,
            ),
            total=len(dataset.encoded_parents),
            desc="Finding optimal branch lengths",
        ):
            optimal_lengths.append(
                self._find_optimal_branch_length(
                    encoded_parent,
                    mask,
                    mutation_indicator,
                    wt_base_modifier,
                    starting_branch_length,
                    **optimization_kwargs,
                )
            )

        self.model.unfreeze()

        return torch.tensor(optimal_lengths)

    def write_loss(self, loss_name, loss, step):
        rate_loss, csp_loss = loss.unbind()
        self.writer.add_scalar("Rate " + loss_name, rate_loss.item(), step)
        self.writer.add_scalar("CSP " + loss_name, csp_loss.item(), step)

def burrito_class_of_model(model):
    if isinstance(model, models.RSCNNModel):
        return RSSHMBurrito
    else:
        return SHMBurrito
