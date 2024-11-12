"""Defining the deep natural selection model (DNSM)."""

import copy

import torch
import torch.nn.functional as F

from netam.common import (
    clamp_probability,
)
from netam.dxsm import DXSMDataset, DXSMBurrito
from netam.hyper_burrito import HyperBurrito
import netam.molevol as molevol
import netam.sequences as sequences


class DNSMDataset(DXSMDataset):

    def update_neutral_probs(self):
        """Update the neutral mutation probabilities for the dataset.

        This is a somewhat vague name, but that's because it includes both the cases of
        the DNSM (in which case it's neutral probabilities of any nonsynonymous
        mutation) and the DASM (in which case it's the neutral probabilities of mutation
        to the various amino acids).

        This is the case of the DNSM, but the DASM will override this method.
        """
        neutral_aa_mut_prob_l = []

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
            if self.multihit_model is not None:
                multihit_model = copy.deepcopy(self.multihit_model).to("cpu")
            else:
                multihit_model = None
            # Note we are replacing all Ns with As, which means that we need to be careful
            # with masking out these positions later. We do this below.
            parent_idxs = sequences.nt_idx_tensor_of_str(nt_parent.replace("N", "A"))
            parent_len = len(nt_parent)
            molevol.check_csps(parent_idxs, nt_csps)

            mut_probs = 1.0 - torch.exp(-branch_length * nt_rates[:parent_len])
            nt_csps = nt_csps[:parent_len, :]

            neutral_aa_mut_prob = molevol.neutral_aa_mut_probs(
                parent_idxs.reshape(-1, 3),
                mut_probs.reshape(-1, 3),
                nt_csps.reshape(-1, 3, 4),
                multihit_model=multihit_model,
            )

            if not torch.isfinite(neutral_aa_mut_prob).all():
                print(f"Found a non-finite neutral_aa_mut_prob")
                print(f"nt_parent: {nt_parent}")
                print(f"mask: {mask}")
                print(f"nt_rates: {nt_rates}")
                print(f"nt_csps: {nt_csps}")
                print(f"branch_length: {branch_length}")
                raise ValueError(
                    f"neutral_aa_mut_prob is not finite: {neutral_aa_mut_prob}"
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
        self.log_neutral_aa_mut_probss = torch.log(torch.stack(neutral_aa_mut_prob_l))

    def __getitem__(self, idx):
        return {
            "aa_parents_idxs": self.aa_parents_idxss[idx],
            "aa_subs_indicator": self.aa_subs_indicators[idx],
            "mask": self.masks[idx],
            "log_neutral_aa_mut_probs": self.log_neutral_aa_mut_probss[idx],
            "nt_rates": self.nt_ratess[idx],
            "nt_csps": self.nt_cspss[idx],
        }

    def to(self, device):
        self.aa_parents_idxss = self.aa_parents_idxss.to(device)
        self.aa_subs_indicators = self.aa_subs_indicators.to(device)
        self.masks = self.masks.to(device)
        self.log_neutral_aa_mut_probss = self.log_neutral_aa_mut_probss.to(device)
        self.nt_ratess = self.nt_ratess.to(device)
        self.nt_cspss = self.nt_cspss.to(device)
        if self.multihit_model is not None:
            self.multihit_model = self.multihit_model.to(device)


class DNSMBurrito(DXSMBurrito):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_branch_lengths(self, in_csv_prefix):
        if self.train_dataset is not None:
            self.train_dataset.load_branch_lengths(
                in_csv_prefix + ".train_branch_lengths.csv"
            )
        self.val_dataset.load_branch_lengths(in_csv_prefix + ".val_branch_lengths.csv")

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
        log_selection_factors = self.model(aa_parents_idxs, mask)
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

    def build_selection_matrix_from_parent(self, parent: str):
        parent = sequences.translate_sequence(parent)
        selection_factors = self.model.selection_factors_of_aa_str(parent)
        selection_matrix = torch.zeros((len(selection_factors), 20), dtype=torch.float)
        # Every "off-diagonal" entry of the selection matrix is set to the selection
        # factor, where "diagonal" means keeping the same amino acid.
        selection_matrix[:, :] = selection_factors[:, None]
        # Set "diagonal" elements to one.
        parent_idxs = sequences.aa_idx_array_of_str(parent)
        selection_matrix[torch.arange(len(parent_idxs)), parent_idxs] = 1.0

        return selection_matrix


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
