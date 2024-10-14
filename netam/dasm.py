"""Here we define a mutation-selection model that is per-amino-acid."""

import torch
import torch.nn.functional as F

# Amazingly, using one thread makes things 50x faster for branch length
# optimization on our server.
torch.set_num_threads(1)

import numpy as np
import pandas as pd

from netam.common import (
    clamp_probability,
)
import netam.dnsm as dnsm
import netam.molevol as molevol
import netam.sequences as sequences
from netam.sequences import (
    translate_sequence,
)


class DASMDataset(dnsm.DNSMDataset):

    def update_neutral_probs(self):
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


def zero_predictions_along_diagonal(predictions, aa_parents_idxs):
    """Zero out the diagonal of a batch of predictions.

    We do this so that we can sum then have the same type of predictions as for the
    DNSM.
    """
    # We would like to do
    # predictions[torch.arange(len(aa_parents_idxs)), aa_parents_idxs] = 0.0
    # but we have a batch dimension. Thus the following.

    batch_size, L, _ = predictions.shape
    batch_indices = torch.arange(batch_size, device=predictions.device)
    predictions[
        batch_indices[:, None],
        torch.arange(L, device=predictions.device),
        aa_parents_idxs,
    ] = 0.0

    return predictions


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
        # Add one entry, zero, to the last dimension of the predictions tensor
        # to handle the ambiguous amino acids. This is the conservative choice.
        # It might be faster to reassign all the 20s to 0s if we are confident
        # in our masking. Perhaps we should always output a 21st dimension
        # for the ambiguous amino acids (see issue #16).
        # If we change something here we should also change the test code
        # in test_dasm.py::test_zero_diagonal.
        predictions = torch.cat(
            [predictions, torch.zeros_like(predictions[:, :, :1])], dim=-1
        )

        predictions = zero_predictions_along_diagonal(predictions, aa_parents_idxs)

        predictions_of_mut = torch.sum(predictions, dim=-1)
        predictions_of_mut = predictions_of_mut.masked_select(mask)
        predictions_of_mut = clamp_probability(predictions_of_mut)
        return self.bce_loss(predictions_of_mut, aa_subs_indicator)

    def build_selection_matrix_from_parent(self, parent: str):
        # This is simpler than the equivalent in dnsm.py because we get the selection
        # matrix directly.
        parent = translate_sequence(parent)
        selection_factors = self.model.selection_factors_of_aa_str(parent)
        parent_idxs = sequences.aa_idx_array_of_str(parent)
        selection_factors[torch.arange(len(parent_idxs)), parent_idxs] = 1.0

        return selection_factors
