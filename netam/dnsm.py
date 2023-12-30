"""
Here we define a mutation-selection model that is just about mutation vs no mutation, and is trainable.

We'll use these conventions:

* B is the batch size
* L is the max sequence length

"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# Amazingly, using one thread makes things 50x faster for branch length
# optimization on our server.
torch.set_num_threads(1)

import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter
from tqdm import tqdm

from epam.torch_common import optimize_branch_length
from epam.models import WrappedBinaryMutSel
import epam.molevol as molevol
import epam.sequences as sequences
from epam.sequences import (
    subs_indicator_tensor_of,
    translate_sequence,
    translate_sequences,
)

from netam.common import (
    MAX_AMBIG_AA_IDX,
    aa_idx_tensor_of_str_ambig,
    clamp_probability,
    mask_tensor_of,
    stack_heterogeneous,
    pick_device,
)
import netam.framework as framework
from netam.hyper_burrito import HyperBurrito


class DNSMDataset(Dataset):
    def __init__(
        self,
        nt_parents,
        nt_children,
        all_rates,
        all_subs_probs,
        branch_length_multiplier=1.0,
    ):
        self.nt_parents = nt_parents
        self.nt_children = nt_children
        self.all_rates = stack_heterogeneous(all_rates.reset_index(drop=True))
        self.all_subs_probs = stack_heterogeneous(all_subs_probs.reset_index(drop=True))

        assert len(self.nt_parents) == len(self.nt_children)
        pcp_count = len(self.nt_parents)

        for parent, child in zip(self.nt_parents, self.nt_children):
            if parent == child:
                raise ValueError(
                    f"Found an identical parent and child sequence: {parent}"
                )

        aa_parents = translate_sequences(self.nt_parents)
        aa_children = translate_sequences(self.nt_children)
        self.max_aa_seq_len = max(len(seq) for seq in aa_parents)
        # We have sequences of varying length, so we start with all tensors set
        # to the ambiguous amino acid, and then will fill in the actual values
        # below.
        self.aa_parents_idxs = torch.full(
            (pcp_count, self.max_aa_seq_len), MAX_AMBIG_AA_IDX
        )
        self.aa_subs_indicator_tensor = torch.zeros((pcp_count, self.max_aa_seq_len))

        self.mask = torch.ones((pcp_count, self.max_aa_seq_len), dtype=torch.bool)

        for i, (aa_parent, aa_child) in enumerate(zip(aa_parents, aa_children)):
            self.mask[i, :] = mask_tensor_of(aa_parent, self.max_aa_seq_len)
            aa_seq_len = len(aa_parent)
            self.aa_parents_idxs[i, :aa_seq_len] = aa_idx_tensor_of_str_ambig(aa_parent)
            self.aa_subs_indicator_tensor[i, :aa_seq_len] = subs_indicator_tensor_of(
                aa_parent, aa_child
            )

        assert torch.all(self.mask.sum(dim=1) > 0)
        assert torch.max(self.aa_parents_idxs) <= MAX_AMBIG_AA_IDX

        # Make initial branch lengths (will get optimized later).
        self._branch_lengths = np.array(
            [
                sequences.mutation_frequency(parent, child) * branch_length_multiplier
                for parent, child in zip(self.nt_parents, self.nt_children)
            ]
        )
        self.update_neutral_aa_mut_probs()

    @property
    def branch_lengths(self):
        return self._branch_lengths

    @branch_lengths.setter
    def branch_lengths(self, new_branch_lengths):
        self._branch_lengths = new_branch_lengths
        self.update_neutral_aa_mut_probs()

    def update_neutral_aa_mut_probs(self):
        print("consolidating shmple rates into substitution probabilities...")

        neutral_aa_mut_prob_l = []

        for nt_parent, mask, rates, branch_length, subs_probs in zip(
            self.nt_parents,
            self.mask,
            self.all_rates,
            self._branch_lengths,
            self.all_subs_probs,
        ):
            mask = mask.to("cpu")
            rates = rates.to("cpu")
            subs_probs = subs_probs.to("cpu")
            # Note we are replacing all Ns with As, which means that we need to be careful
            # with masking out these positions later. We do this below.
            parent_idxs = sequences.nt_idx_tensor_of_str(nt_parent.replace("N", "A"))
            parent_len = len(nt_parent)

            mut_probs = 1.0 - torch.exp(-branch_length * rates[:parent_len])
            normed_subs_probs = molevol.normalize_sub_probs(
                parent_idxs, subs_probs[:parent_len, :]
            )

            neutral_aa_mut_prob = molevol.neutral_aa_mut_probs(
                parent_idxs.reshape(-1, 3),
                mut_probs.reshape(-1, 3),
                normed_subs_probs.reshape(-1, 3, 4),
            )

            # Ensure that all values are positive before taking the log later
            neutral_aa_mut_prob = clamp_probability(neutral_aa_mut_prob)

            pad_len = self.max_aa_seq_len - neutral_aa_mut_prob.shape[0]
            if pad_len > 0:
                neutral_aa_mut_prob = F.pad(
                    neutral_aa_mut_prob, (0, pad_len), value=1e-8
                )
            # Here we zero out masked positions.
            neutral_aa_mut_prob *= mask

            neutral_aa_mut_prob_l.append(neutral_aa_mut_prob)

        # Note that our masked out positions will have a nan log probability,
        # which will require us to handle them correctly downstream.
        self.log_neutral_aa_mut_probs = torch.log(torch.stack(neutral_aa_mut_prob_l))

    def __len__(self):
        return len(self.aa_parents_idxs)

    def __getitem__(self, idx):
        return {
            "aa_parents_idxs": self.aa_parents_idxs[idx],
            "subs_indicator": self.aa_subs_indicator_tensor[idx],
            "mask": self.mask[idx],
            "log_neutral_aa_mut_probs": self.log_neutral_aa_mut_probs[idx],
            "rates": self.all_rates[idx],
            "subs_probs": self.all_subs_probs[idx],
        }

    def to(self, device):
        self.aa_parents_idxs = self.aa_parents_idxs.to(device)
        self.aa_subs_indicator_tensor = self.aa_subs_indicator_tensor.to(device)
        self.mask = self.mask.to(device)
        self.log_neutral_aa_mut_probs = self.log_neutral_aa_mut_probs.to(device)
        self.all_rates = self.all_rates.to(device)
        self.all_subs_probs = self.all_subs_probs.to(device)


def train_test_datasets_of_pcp_df(pcp_df, train_frac=0.8, branch_length_multiplier=1.0):
    nt_parents = pcp_df["parent"].reset_index(drop=True)
    nt_children = pcp_df["child"].reset_index(drop=True)
    rates = pcp_df["rates"].reset_index(drop=True)
    subs_probs = pcp_df["subs_probs"].reset_index(drop=True)

    train_len = int(train_frac * len(nt_parents))
    train_parents, val_parents = nt_parents[:train_len], nt_parents[train_len:]
    train_children, val_children = nt_children[:train_len], nt_children[train_len:]
    train_rates, val_rates = rates[:train_len], rates[train_len:]
    train_subs_probs, val_subs_probs = (
        subs_probs[:train_len],
        subs_probs[train_len:],
    )

    train_dataset = DNSMDataset(
        train_parents,
        train_children,
        train_rates,
        train_subs_probs,
        branch_length_multiplier=branch_length_multiplier,
    )
    val_dataset = DNSMDataset(
        val_parents,
        val_children,
        val_rates,
        val_subs_probs,
        branch_length_multiplier=branch_length_multiplier,
    )

    return train_dataset, val_dataset


class DNSMBurrito(framework.Burrito):
    def __init__(self, *args, device=pick_device(), **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.model.to(self.device)
        self.wrapped_model = WrappedBinaryMutSel(self.model, weights_directory=None)

    def loss_of_batch(self, batch):
        aa_parents_idxs = batch["aa_parents_idxs"].to(self.device)
        aa_subs_indicator = batch["subs_indicator"].to(self.device)
        mask = batch["mask"].to(self.device)
        log_neutral_aa_mut_probs = batch["log_neutral_aa_mut_probs"].to(self.device)
        log_selection_factors = self.model(aa_parents_idxs, mask)
        return self.complete_loss_fn(
            log_neutral_aa_mut_probs,
            log_selection_factors,
            aa_subs_indicator,
            mask,
        )

    def complete_loss_fn(
        self,
        log_neutral_aa_mut_probs,
        log_selection_factors,
        aa_subs_indicator,
        mask,
    ):
        # Take the product of the neutral mutation probabilities and the selection factors.
        predictions = torch.exp(log_neutral_aa_mut_probs + log_selection_factors)

        predictions = predictions.masked_select(mask)
        aa_subs_indicator = aa_subs_indicator.masked_select(mask)

        # Because the neutral mutation probabilities should be normalized, and
        # log_selection_factors comes from a logsigmoid, we should have
        # predictions <= 1.
        assert torch.all(predictions <= 1)

        return self.bce_loss(predictions, aa_subs_indicator)

    # TODO deprecated
    def _build_log_pcp_probability(
        self, parent: str, child: str, rates: Tensor, subs_probs: Tensor
    ):
        """
        This version of _build_log_pcp_probability directly expresses BCELoss
        so that we're minimizing the same loss as the NN when we're optimizing
        branch length.
        """

        assert len(parent) % 3 == 0

        # Note we are replacing all Ns with As, which means that we need to be careful
        # with masking out these positions later. We do this below.
        parent_idxs = sequences.nt_idx_tensor_of_str(parent.replace("N", "A"))
        aa_parent = translate_sequence(parent)
        aa_child = translate_sequence(child)
        aa_subs_indicator = subs_indicator_tensor_of(aa_parent, aa_child)
        mask = mask_tensor_of(aa_parent)
        selection_factors = self.model.selection_factors_of_aa_str(aa_parent).to("cpu")
        rates = rates.to("cpu")
        subs_probs = subs_probs.to("cpu")
        bce_loss = nn.BCELoss()

        def log_pcp_probability(log_branch_length: torch.Tensor):
            branch_length = torch.exp(log_branch_length)
            mut_probs = 1.0 - torch.exp(-branch_length * rates)
            normed_subs_probs = molevol.normalize_sub_probs(parent_idxs, subs_probs)

            neutral_aa_mut_prob = molevol.neutral_aa_mut_probs(
                parent_idxs.reshape(-1, 3),
                mut_probs.reshape(-1, 3),
                normed_subs_probs.reshape(-1, 3, 4),
            )

            predictions = neutral_aa_mut_prob * selection_factors
            predictions = predictions.masked_select(mask)
            assert torch.all((predictions >= 0) & (predictions <= 1))
            masked_indicator = aa_subs_indicator.masked_select(mask)
            # Negative because BCELoss is negative log likelihood.
            return -bce_loss(predictions, masked_indicator)

        return log_pcp_probability

    def _find_optimal_branch_length(
        self, parent, child, rates, subs_probs, starting_branch_length
    ):
        if parent == child:
            return 0.0
        log_pcp_probability = self.wrapped_model._build_log_pcp_probability(
            parent, child, rates, subs_probs
        )
        # TODO do we want to add some optimization options here?
        return optimize_branch_length(log_pcp_probability, starting_branch_length)

    def find_optimal_branch_lengths(
        self,
        nt_parents,
        nt_children,
        all_rates,
        all_subs_probs,
        starting_branch_lengths,
    ):
        optimal_lengths = []

        for parent, child, rates, subs_probs, starting_length in tqdm(
            zip(
                nt_parents,
                nt_children,
                all_rates,
                all_subs_probs,
                starting_branch_lengths,
            ),
            total=len(nt_parents),
            desc="Finding optimal branch lengths",
        ):
            optimal_lengths.append(
                self._find_optimal_branch_length(
                    parent,
                    child,
                    rates[: len(parent)],
                    subs_probs[: len(parent), :],
                    starting_length,
                )
            )

        return np.array(optimal_lengths)

    def optimize_branch_lengths(self):
        # We do the branch length optimization but want to restore the model to
        # the device it was on before.
        device = next(self.model.parameters()).device
        self.model.to("cpu")
        for dataset in [self.train_loader.dataset, self.val_loader.dataset]:
            # make an assertion that dataset.all_rates is on the cpu
            assert dataset.all_rates.device.type == "cpu"
            dataset.branch_lengths = self.find_optimal_branch_lengths(
                dataset.nt_parents,
                dataset.nt_children,
                dataset.all_rates,
                dataset.all_subs_probs,
                dataset.branch_lengths,
            )
        self.model.to(device)

    def joint_train(self, epochs=20, cycle_count=2):
        """
        Do joint optimization of model and branch lengths.
        """
        loss_history_l = []
        loss_history_l.append(self.train(3))
        self.optimize_branch_lengths()
        # TODO do we still need to do this?
        # We double branch lengths and then retrain to avoid the best parameter
        # values hitting the boundary of 1.
        self.train_loader.dataset.branch_lengths *= 2
        self.val_loader.dataset.branch_lengths *= 2
        self.reset_optimization()
        for cycle in range(cycle_count):
            loss_history_l.append(self.train(epochs))
            if cycle < cycle_count - 1:
                self.optimize_branch_lengths()
        return pd.concat(loss_history_l, ignore_index=True)

    def full_train(self, epochs=100):
        """
        For now, just optimize the model and not branch lengths.
        """
        return self.train(epochs)

    def to_crepe(self):
        training_hyperparameters = {
            key: self.__dict__[key]
            for key in [
                "learning_rate",
            ]
        }
        encoder = framework.PlaceholderEncoder()
        return framework.Crepe(encoder, self.model, training_hyperparameters)


class DNSMHyperBurrito(HyperBurrito):
    # Note that we have to write the args out explicitly because we use some magic to filter kwargs in the optuna_objective method.
    def burrito_of_model(
        self,
        model,
        device,
        batch_size=1024,
        learning_rate=0.1,
        min_learning_rate=1e-4,
        l2_regularization_coeff=1e-6,
        verbose=False,
    ):
        burrito = DNSMBurrito(
            self.train_dataset,
            self.val_dataset,
            model,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            l2_regularization_coeff=l2_regularization_coeff,
            verbose=verbose,
        )
        return burrito
