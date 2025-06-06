"""Here we define a model that outputs a vector of 20 amino acid preferences."""

import torch
import torch.nn.functional as F

from netam.common import clamp_probability
from netam.dxsm import DXSMDataset, DXSMBurrito, zap_predictions_along_diagonal
import netam.framework as framework
import netam.molevol as molevol
import netam.sequences as sequences
from typing import Tuple


class DDSMDataset(DXSMDataset):

    def update_neutral_probs(self):
        neutral_aa_probs_light = []

        for nt_parent, mask, nt_rates, nt_csps, branch_length in zip(
            self.nt_parents,
            self.masks,
            self.nt_ratess,
            self.nt_cspss,
            self.branch_lengths,
        ):
            # Note we are replacing all Ns with As, which means that we need to be careful
            # with masking out these positions later. We do this below.
            parent_idxs = sequences.nt_idx_tensor_of_str(nt_parent.replace("N", "A"))
            parent_len = len(nt_parent)

            mut_probs = 1.0 - torch.exp(-branch_length * nt_rates[:parent_len])
            nt_csps = nt_csps[:parent_len, :]
            nt_mask = mask.repeat_interleave(3)[: len(nt_parent)]
            molevol.check_csps(parent_idxs[nt_mask], nt_csps[: len(nt_parent)][nt_mask])

            neutral_aa_probs = molevol.neutral_aa_probs(
                parent_idxs.reshape(-1, 3),
                mut_probs.reshape(-1, 3),
                nt_csps.reshape(-1, 3, 4),
                multihit_model=self.multihit_model,
            )

            if not torch.isfinite(neutral_aa_probs).all():
                print("Found a non-finite neutral_aa_probs")
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

            neutral_aa_probs_light.append(neutral_aa_probs)

        # Note that our masked out positions will have a nan log probability,
        # which will require us to handle them correctly downstream.
        self.log_neutral_aa_probss = torch.log(torch.stack(neutral_aa_probs_light))

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

    def move_data_to_device(self, device):
        self.aa_parents_idxss = self.aa_parents_idxss.to(device)
        self.aa_children_idxss = self.aa_children_idxss.to(device)
        self.aa_subs_indicators = self.aa_subs_indicators.to(device)
        self.masks = self.masks.to(device)
        self.log_neutral_aa_probss = self.log_neutral_aa_probss.to(device)
        self.nt_ratess = self.nt_ratess.to(device)
        self.nt_cspss = self.nt_cspss.to(device)
        if self.multihit_model is not None:
            self.multihit_model = self.multihit_model.to(device)


class DDSMBurrito(framework.TwoLossMixin, DXSMBurrito):
    model_type = "ddsm"

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
        log_selection_factors = self.selection_factors_of_aa_idxs(aa_parents_idxs, mask)
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
        subs_mask = (aa_subs_indicator == 1) & mask
        csp_pred = predictions[subs_mask]
        csp_targets = aa_children_idxs[subs_mask]
        csp_loss = self.xent_loss(csp_pred, csp_targets)
        return torch.stack([subs_pos_loss, csp_loss])

    # This is not used anywhere, except for in a few tests. Keeping it around
    # for that reason.
    def _build_selection_matrix_from_parent(self, parent: Tuple[str, str]):
        """Build a selection matrix from a parent nucleotide sequence, a heavy-chain,
        light-chain pair.

        Values at ambiguous sites are meaningless. Returned value is a tuple of
        selection matrix for heavy and light chain sequences.
        """
        # This is simpler than the equivalent in dnsm.py because we get the selection
        # matrix directly. Note that selection_factors_of_aa_str does the exponentiation
        # so this indeed gives us the selection factors, not the log selection factors.
        aa_parent_pair = tuple(map(sequences.translate_sequence, parent))
        per_aa_selection_factorss = self.model.selection_factors_of_aa_str(
            aa_parent_pair
        )

        result = []
        for per_aa_selection_factors, aa_parent in zip(
            per_aa_selection_factorss, aa_parent_pair
        ):
            aa_parent_idxs = torch.tensor(sequences.aa_idx_array_of_str(aa_parent))
            if len(per_aa_selection_factors) > 0:
                result.append(
                    zap_predictions_along_diagonal(
                        per_aa_selection_factors.unsqueeze(0),
                        aa_parent_idxs.unsqueeze(0),
                        fill=1.0,
                    ).squeeze(0)
                )
            else:
                result.append(per_aa_selection_factors)

        return tuple(result)
