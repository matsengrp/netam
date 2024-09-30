"""Here we define a mutation-selection model that is per-amino-acid."""

import copy
import multiprocessing as mp

import torch
import torch.nn.functional as F

# Amazingly, using one thread makes things 50x faster for branch length
# optimization on our server.
torch.set_num_threads(1)

import numpy as np
import pandas as pd

from tqdm import tqdm

from netam.common import (
    MAX_AMBIG_AA_IDX,
    aa_idx_tensor_of_str_ambig,
    clamp_probability,
    aa_mask_tensor_of,
    stack_heterogeneous,
)
import netam.dnsm as dnsm
import netam.framework as framework
from netam.hyper_burrito import HyperBurrito
import netam.molevol as molevol
import netam.sequences as sequences
from netam.sequences import (
    aa_subs_indicator_tensor_of,
    translate_sequence,
    translate_sequences,
)


class DASMDataset(dnsm.DNSMDataset):

    # TODO should we rename this?
    def update_neutral_aa_mut_probs(self):
        neutral_aa_probs_l = []

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

            neutral_aa_probs = molevol.neutral_aa_probs(
                parent_idxs.reshape(-1, 3),
                mut_probs.reshape(-1, 3),
                normed_subs_probs.reshape(-1, 3, 4),
            )

            if not torch.isfinite(neutral_aa_probs).all():
                print(f"Found a non-finite neutral_aa_probs")
                print(f"nt_parent: {nt_parent}")
                print(f"mask: {mask}")
                print(f"rates: {rates}")
                print(f"subs_probs: {subs_probs}")
                print(f"branch_length: {branch_length}")
                raise ValueError(f"neutral_aa_probs is not finite: {neutral_aa_probs}")

            # Ensure that all values are positive before taking the log later
            neutral_aa_probs = clamp_probability(neutral_aa_probs)

            pad_len = self.max_aa_seq_len - neutral_aa_probs.shape[0]
            if pad_len > 0:
                neutral_aa_probs = F.pad(
                    neutral_aa_probs, (0, 0, 0, pad_len), value=1e-8
                )
            # Here we zero out masked positions.
            neutral_aa_probs *= mask[:, None]

            neutral_aa_probs_l.append(neutral_aa_probs)

        # Note that our masked out positions will have a nan log probability,
        # which will require us to handle them correctly downstream.
        self.log_neutral_aa_probs = torch.log(torch.stack(neutral_aa_probs_l))

    def __getitem__(self, idx):
        return {
            "aa_parents_idxs": self.aa_parents_idxs[idx],
            "subs_indicator": self.aa_subs_indicator_tensor[idx],
            "mask": self.mask[idx],
            "log_neutral_aa_probs": self.log_neutral_aa_probs[idx],
            "rates": self.all_rates[idx],
            "subs_probs": self.all_subs_probs[idx],
        }

    def to(self, device):
        self.aa_parents_idxs = self.aa_parents_idxs.to(device)
        self.aa_subs_indicator_tensor = self.aa_subs_indicator_tensor.to(device)
        self.mask = self.mask.to(device)
        self.log_neutral_aa_probs = self.log_neutral_aa_probs.to(device)
        self.all_rates = self.all_rates.to(device)
        self.all_subs_probs = self.all_subs_probs.to(device)


# TODO second step. code dup: class method as in dnsm.py
def train_val_datasets_of_pcp_df(pcp_df, branch_length_multiplier=5.0):
    """Perform a train-val split based on a "in_train" column.

    Stays here so it can be used in tests.
    """
    train_df = pcp_df[pcp_df["in_train"]].reset_index(drop=True)
    val_df = pcp_df[~pcp_df["in_train"]].reset_index(drop=True)
    val_dataset = DASMDataset.of_pcp_df(
        val_df, branch_length_multiplier=branch_length_multiplier
    )
    if len(train_df) == 0:
        return None, val_dataset
    # else:
    train_dataset = DASMDataset.of_pcp_df(
        train_df, branch_length_multiplier=branch_length_multiplier
    )
    return train_dataset, val_dataset


class DASMBurrito(dnsm.DNSMBurrito):

    def prediction_pair_of_batch(self, batch):
        """Get log neutral AA probabilities and log selection factors for a batch of
        data."""
        aa_parents_idxs = batch["aa_parents_idxs"].to(self.device)
        mask = batch["mask"].to(self.device)
        log_neutral_aa_probs = batch["log_neutral_aa_probs"].to(self.device)
        if not torch.isfinite(log_neutral_aa_probs[mask]).all():
            raise ValueError(
                f"log_neutral_aa_probs has non-finite values at relevant positions: {log_neutral_aa_probs[mask]}"
            )
        log_selection_factors = self.model(aa_parents_idxs, mask)
        return log_neutral_aa_probs, log_selection_factors

    def predictions_of_pair(self, log_neutral_aa_probs, log_selection_factors):
        # Take the product of the neutral mutation probabilities and the selection factors.
        # NOTE each of these now have last dimension of 20
        # this is p_{j, a} * f_{j, a} 
        predictions = torch.exp(log_neutral_aa_probs + log_selection_factors)
        assert torch.isfinite(predictions).all()
        predictions = clamp_probability(predictions)
        return predictions

    def predictions_of_batch(self, batch):
        """Make predictions for a batch of data.

        Note that we use the mask for prediction as part of the input for the
        transformer, though we don't mask the predictions themselves.
        """
        log_neutral_aa_probs, log_selection_factors = self.prediction_pair_of_batch(
            batch
        )
        return self.predictions_of_pair(log_neutral_aa_probs, log_selection_factors)

    def loss_of_batch(self, batch):
        aa_subs_indicator = batch["subs_indicator"].to(self.device)
        mask = batch["mask"].to(self.device)
        aa_parents_idxs = batch["aa_parents_idxs"].to(self.device)
        aa_subs_indicator = aa_subs_indicator.masked_select(mask)
        predictions = self.predictions_of_batch(batch)
        # add one entry, zero, to the last dimension of the predictions tensor
        # to handle the ambiguous amino acids
        # TODO perhaps we can do better: perhaps we can be confident in our masking that is going to take care of this if we re assign all the 20s to 0s.
        # OR should we just always output a 21st dimension?
        predictions = torch.cat(
            [predictions, torch.zeros_like(predictions[:, :, :1])], dim=-1
        )
        # Now we make predictions of mutation by taking everything off the diagonal.
        # We would like to do
        # predictions[torch.arange(len(aa_parents_idxs)), aa_parents_idxs] = 0.0
        # but we have a batch dimension. Thus the following.

        # Get batch size and sequence length
        batch_size, L, _ = predictions.shape
        # Create indices for each batch
        batch_indices = torch.arange(batch_size, device=self.device)
        # Zero out the diagonal by setting predictions[batch_idx, site_idx, aa_idx] to 0
        # TODO play around with this in the notebook? Or just print things?
        predictions[
            batch_indices[:, None], torch.arange(L, device=self.device), aa_parents_idxs
        ] = 0.0

        predictions_of_mut = torch.sum(predictions, dim=-1)
        predictions_of_mut = predictions_of_mut.masked_select(mask)
        return self.bce_loss(predictions_of_mut, aa_subs_indicator)

    def build_selection_matrix_from_parent(self, parent: str):
        # This is simpler than the equivalent in dnsm.py because we get the selection
        # matrix directly.
        parent = translate_sequence(parent)
        selection_factors = self.model.selection_factors_of_aa_str(parent)
        parent_idxs = sequences.aa_idx_array_of_str(parent)
        selection_factors[torch.arange(len(parent_idxs)), parent_idxs] = 1.0

        return selection_factors

    # We need to repeat this so that we use this worker_optimize_branch_length below.
    def find_optimal_branch_lengths(self, dataset, **optimization_kwargs):
        worker_count = min(mp.cpu_count() // 2, 10)
        # The following can be used when one wants a better traceback.
        # burrito = DNSMBurrito(None, dataset, copy.deepcopy(self.model))
        # return burrito.serial_find_optimal_branch_lengths(dataset, **optimization_kwargs)
        with mp.Pool(worker_count) as pool:
            splits = dataset.split(worker_count)
            results = pool.starmap(
                worker_optimize_branch_length,
                [(self.model, split, optimization_kwargs) for split in splits],
            )
        return torch.cat(results)


def worker_optimize_branch_length(model, dataset, optimization_kwargs):
    """The worker used for parallel branch length optimization."""
    burrito = DASMBurrito(None, dataset, copy.deepcopy(model))
    return burrito.serial_find_optimal_branch_lengths(dataset, **optimization_kwargs)
