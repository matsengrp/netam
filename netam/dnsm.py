"""Defining the deep natural selection model (DNSM)."""

import torch
import torch.nn.functional as F

from netam.common import (
    clamp_probability,
)
from netam.dxsm import DXSMDataset, DXSMBurrito
from netam.hyper_burrito import HyperBurrito
import netam.molevol as molevol
import netam.sequences as sequences

from typing import Tuple


class DNSMDataset(DXSMDataset):

    def update_neutral_probs(self):
        """Update the neutral mutation probabilities for the dataset.

        This is a somewhat vague name, but that's because it includes both the cases of
        the DNSM (in which case it's neutral probabilities of any nonsynonymous
        mutation) and the DDSM (in which case it's the neutral probabilities of mutation
        to the various amino acids).

        This is the case of the DNSM, but the DDSM will override this method.
        """
        neutral_aa_mut_prob_light = []

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
            # Cannot assume that nt_csps and mask are same length, because when
            # datasets are split, masks are recomputed.
            nt_mask = mask.repeat_interleave(3)[:parent_len]
            molevol.check_csps(parent_idxs[nt_mask], nt_csps[:parent_len][nt_mask])

            nt_csps = nt_csps[:parent_len, :]

            neutral_aa_mut_probs = molevol.non_stop_neutral_aa_mut_probs(
                parent_idxs,
                nt_rates[:parent_len],
                nt_csps,
                branch_length,
                multihit_model=self.multihit_model,
            )

            if not torch.isfinite(neutral_aa_mut_probs).all():
                print("Found a non-finite neutral_aa_mut_prob")
                print(f"nt_parent: {nt_parent}")
                print(f"mask: {mask}")
                print(f"nt_rates: {nt_rates}")
                print(f"nt_csps: {nt_csps}")
                print(f"branch_length: {branch_length}")
                raise ValueError(
                    f"neutral_aa_mut_prob is not finite: {neutral_aa_mut_probs}"
                )

            # Ensure that all values are positive before taking the log later
            neutral_aa_mut_probs = clamp_probability(neutral_aa_mut_probs)

            pad_len = self.max_aa_seq_len - neutral_aa_mut_probs.shape[0]
            if pad_len > 0:
                neutral_aa_mut_probs = F.pad(
                    neutral_aa_mut_probs, (0, pad_len), value=1e-8
                )
            # Here we zero out masked positions.
            neutral_aa_mut_probs *= mask

            neutral_aa_mut_prob_light.append(neutral_aa_mut_probs)

        # Note that our masked out positions will have a nan log probability,
        # which will require us to handle them correctly downstream.
        self.log_neutral_aa_mut_probss = torch.log(
            torch.stack(neutral_aa_mut_prob_light)
        )

    def __getitem__(self, idx):
        return {
            "aa_parents_idxs": self.aa_parents_idxss[idx],
            "aa_subs_indicator": self.aa_subs_indicators[idx],
            "mask": self.masks[idx],
            "log_neutral_aa_mut_probs": self.log_neutral_aa_mut_probss[idx],
            "nt_rates": self.nt_ratess[idx],
            "nt_csps": self.nt_cspss[idx],
        }

    def move_data_to_device(self, device):
        self.aa_parents_idxss = self.aa_parents_idxss.to(device)
        self.aa_subs_indicators = self.aa_subs_indicators.to(device)
        self.masks = self.masks.to(device)
        self.log_neutral_aa_mut_probss = self.log_neutral_aa_mut_probss.to(device)
        self.nt_ratess = self.nt_ratess.to(device)
        self.nt_cspss = self.nt_cspss.to(device)
        if self.multihit_model is not None:
            self.multihit_model = self.multihit_model.to(device)


class DNSMBurrito(DXSMBurrito):

    model_type = "dnsm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prediction_pair_of_batch(self, batch):
        """Get log neutral amino acid substitution probabilities and log selection
        factors for a batch of data."""
        aa_parents_idxs = batch["aa_parents_idxs"].to(self.device)
        mask = batch["mask"].to(self.device)
        log_neutral_aa_mut_probs = batch["log_neutral_aa_mut_probs"].to(self.device)
        if not torch.isfinite(log_neutral_aa_mut_probs[mask]).all():
            raise ValueError(
                f"log_neutral_aa_mut_probs has non-finite values at relevant positions: {log_neutral_aa_mut_probs[mask]}"
            )
        # Right here is where model is evaluated!
        log_selection_factors = self.selection_factors_of_aa_idxs(aa_parents_idxs, mask)
        return log_neutral_aa_mut_probs, log_selection_factors

    def predictions_of_pair(self, log_neutral_aa_mut_probs, log_selection_factors):
        """Obtain the predictions for a pair consisting of the log neutral amino acid
        mutation substitution probabilities and the log selection factors."""
        predictions = torch.exp(log_neutral_aa_mut_probs + log_selection_factors)
        assert torch.isfinite(predictions).all()
        predictions = clamp_probability(predictions)
        return predictions

    def predictions_of_batch(self, batch):
        """Make predictions for a batch of data.

        Note that we use the mask for prediction as part of the input for the
        transformer, though we don't mask the predictions themselves.
        """
        log_neutral_aa_mut_probs, log_selection_factors = self.prediction_pair_of_batch(
            batch
        )
        return self.predictions_of_pair(log_neutral_aa_mut_probs, log_selection_factors)

    def loss_of_batch(self, batch):
        aa_subs_indicator = batch["aa_subs_indicator"].to(self.device)
        mask = batch["mask"].to(self.device)
        aa_subs_indicator = aa_subs_indicator.masked_select(mask)
        predictions = self.predictions_of_batch(batch).masked_select(mask)
        return self.bce_loss(predictions, aa_subs_indicator)

    def _build_selection_matrix_from_selection_factors(
        self, selection_factors, aa_parent_idxs
    ):
        """Build a selection matrix from a selection factor tensor for a single
        sequence.

        upgrades the provided tensor containing a selection factor per site to a matrix
        containing a selection factor per site and amino acid. The wildtype aa selection
        factor is set to 1, and the rest are set to the selection factor.
        """
        return molevol.lift_to_per_aa_selection_factors(
            selection_factors, aa_parent_idxs
        )

    def build_selection_matrix_from_parent_aa(
        self, aa_parent_idxs: torch.Tensor, mask: torch.Tensor
    ):
        """Build a selection matrix from a single parent amino acid sequence.

        Values at ambiguous sites are meaningless.
        """
        with torch.no_grad():
            selection_factors = (
                self.selection_factors_of_aa_idxs(
                    aa_parent_idxs.unsqueeze(0), mask.unsqueeze(0)
                )
                .squeeze(0)
                .exp()
            )
        return self._build_selection_matrix_from_selection_factors(
            selection_factors, aa_parent_idxs
        )

    def _build_selection_matrix_from_parent(self, parent: Tuple[str, str]):
        """Build a selection matrix from a nucleotide sequence.

        Values at ambiguous sites are meaningless.
        """
        aa_parent_pair = tuple(map(sequences.translate_sequence, parent))
        selection_factorss = self.model.selection_factors_of_aa_str(aa_parent_pair)

        result = []
        for selection_factors, aa_parent in zip(selection_factorss, aa_parent_pair):
            aa_parent_idxs = sequences.aa_idx_array_of_str(aa_parent)
            if len(selection_factors) > 0:
                result.append(
                    self._build_selection_matrix_from_selection_factors(
                        selection_factors, aa_parent_idxs
                    )
                )
            else:
                result.append(torch.empty(0, 20))
        return tuple(result)


class DNSMHyperBurrito(HyperBurrito):
    # Note that we have to write the args out explicitly because we use some magic to filter kwargs in the optuna_objective method.
    def burrito_of_model(
        self,
        model,
        device,
        batch_size=1024,
        learning_rate=0.1,
        min_learning_rate=1e-4,
        weight_decay=1e-6,
    ):
        model.to(device)
        burrito = DNSMBurrito(
            self.train_dataset,
            self.val_dataset,
            model,
            batch_size=batch_size,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            weight_decay=weight_decay,
        )
        return burrito
