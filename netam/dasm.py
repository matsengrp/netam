"""Here we define a model that outputs a vector of 20 amino acid preferences."""

import torch
import torch.nn.functional as F

from netam.common import (
    clamp_probability,
    BIG,
)
from netam.dxsm import DXSMDataset, DXSMBurrito
import netam.framework as framework
import netam.molevol as molevol
import netam.sequences as sequences


class DASMDataset(DXSMDataset):

    def update_neutral_probs(self):
        neutral_aa_probs_l = []

        for nt_parent, mask, nt_rates, nt_csps, branch_length in zip(
            self.nt_parents,
            self.masks,
            self.nt_ratess,
            self.nt_cspss,
            self._branch_lengths,
        ):
            mask = mask.to("cpu")
            nt_rates = nt_rates.to("cpu")
            nt_csps = nt_csps.to("cpu")
            # Note we are replacing all Ns with As, which means that we need to be careful
            # with masking out these positions later. We do this below.
            parent_idxs = sequences.nt_idx_tensor_of_str(nt_parent.replace("N", "A"))
            parent_len = len(nt_parent)

            mut_probs = 1.0 - torch.exp(-branch_length * nt_rates[:parent_len])
            nt_csps = nt_csps[:parent_len, :]
            molevol.check_csps(parent_idxs, nt_csps)

            neutral_aa_probs = molevol.neutral_aa_probs(
                parent_idxs.reshape(-1, 3),
                mut_probs.reshape(-1, 3),
                nt_csps.reshape(-1, 3, 4),
            )

            if not torch.isfinite(neutral_aa_probs).all():
                print(f"Found a non-finite neutral_aa_probs")
                print(f"nt_parent: {nt_parent}")
                print(f"mask: {mask}")
                print(f"nt_rates: {nt_rates}")
                print(f"nt_csps: {nt_csps}")
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
        self.log_neutral_aa_probss = torch.log(torch.stack(neutral_aa_probs_l))

    def __getitem__(self, idx):
        return {
            "aa_parents_idxs": self.aa_parents_idxss[idx],
            "aa_children_idxs": self.aa_children_idxss[idx],
            "subs_indicator": self.aa_subs_indicators[idx],
            "mask": self.masks[idx],
            "log_neutral_aa_probs": self.log_neutral_aa_probss[idx],
            "nt_rates": self.nt_ratess[idx],
            "nt_csps": self.nt_cspss[idx],
        }

    def to(self, device):
        self.aa_parents_idxss = self.aa_parents_idxss.to(device)
        self.aa_children_idxss = self.aa_children_idxss.to(device)
        self.aa_subs_indicators = self.aa_subs_indicators.to(device)
        self.masks = self.masks.to(device)
        self.log_neutral_aa_probss = self.log_neutral_aa_probss.to(device)
        self.nt_ratess = self.nt_ratess.to(device)
        self.nt_cspss = self.nt_cspss.to(device)
        if self.multihit_model is not None:
            self.multihit_model = self.multihit_model.to(device)


def zap_predictions_along_diagonal(predictions, aa_parents_idxs):
    """Set the diagonal (i.e. no amino acid change) of the predictions tensor to -BIG,
    except where aa_parents_idxs >= 20, which indicates no update should be done."""

    device = predictions.device
    batch_size, L, _ = predictions.shape
    batch_indices = torch.arange(batch_size, device=device)[:, None].expand(-1, L)
    sequence_indices = torch.arange(L, device=device)[None, :].expand(batch_size, -1)

    # Create a mask for valid positions (where aa_parents_idxs is less than 20)
    valid_mask = aa_parents_idxs < 20

    # Only update the predictions for valid positions
    predictions[
        batch_indices[valid_mask],
        sequence_indices[valid_mask],
        aa_parents_idxs[valid_mask],
    ] = -BIG

    return predictions


class DASMBurrito(framework.TwoLossMixin, DXSMBurrito):
    def __init__(self, *args, loss_weights: list = [1.0, 0.01], **kwargs):
        super().__init__(*args, **kwargs)
        self.xent_loss = torch.nn.CrossEntropyLoss()
        self.loss_weights = torch.tensor(loss_weights).to(self.device)

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
        """Take the sum of the neutral mutation log probabilities and the selection
        factors.

        In contrast to a DNSM, each of these now have last dimension of 20.
        """
        predictions = log_neutral_aa_probs + log_selection_factors
        assert torch.isnan(predictions).sum() == 0
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
        # Netam issue #16: child mask would be preferable here.
        mask = batch["mask"].to(self.device)
        aa_parents_idxs = batch["aa_parents_idxs"].to(self.device)
        aa_children_idxs = batch["aa_children_idxs"].to(self.device)
        masked_aa_subs_indicator = aa_subs_indicator.masked_select(mask)
        predictions = self.predictions_of_batch(batch)

        # "Zapping" out the diagonal means setting it to zero in log space by
        # setting it to -BIG. This is a no-op for sites that have an X
        # (ambiguous AA) in the parent. This could cause problems in principle,
        # but in practice we mask out sites with Xs in the parent for the
        # mut_pos_loss, and we mask out sites with no substitution for the CSP
        # loss. The latter class of sites also eliminates sites that have Xs in
        # the parent or child (see sequences.aa_subs_indicator_tensor_of).
        predictions = zap_predictions_along_diagonal(predictions, aa_parents_idxs)

        # After zapping out the diagonal, we can effectively sum over the
        # off-diagonal elements to get the probability of a nonsynonymous
        # substitution.
        subs_pos_pred = torch.sum(torch.exp(predictions), dim=-1)
        subs_pos_pred = subs_pos_pred.masked_select(mask)
        subs_pos_pred = clamp_probability(subs_pos_pred)
        subs_pos_loss = self.bce_loss(subs_pos_pred, masked_aa_subs_indicator)

        # We now need to calculate the conditional substitution probability
        # (CSP) loss. We have already zapped out the diagonal, and we're in
        # logit space, so we are set up for using the cross entropy loss.
        # However we have to mask out the sites that are not substituted, i.e.
        # the sites for which aa_subs_indicator is 0.
        subs_mask = aa_subs_indicator == 1
        csp_pred = predictions[subs_mask]
        csp_targets = aa_children_idxs[subs_mask]
        csp_loss = self.xent_loss(csp_pred, csp_targets)

        return torch.stack([subs_pos_loss, csp_loss])

    def build_selection_matrix_from_parent(self, parent: str):
        # This is simpler than the equivalent in dnsm.py because we get the selection
        # matrix directly. Note that selection_factors_of_aa_str does the exponentiation
        # so this indeed gives us the selection factors, not the log selection factors.
        parent = sequences.translate_sequence(parent)
        per_aa_selection_factors = self.model.selection_factors_of_aa_str(parent)
        parent_idxs = sequences.aa_idx_array_of_str(parent)
        per_aa_selection_factors[torch.arange(len(parent_idxs)), parent_idxs] = 1.0

        return per_aa_selection_factors
