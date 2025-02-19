"""Defining the deep natural selection model (DNSM)."""

import torch
import torch.nn.functional as F

from netam.common import (
    clamp_probability,
    BIG,
)
from netam.dxsm import DXSMDataset, DXSMBurrito
import netam.molevol as molevol

from netam.sequences import (
    nt_idx_tensor_of_str,
    codon_idx_tensor_of_str_ambig,
    AMBIGUOUS_CODON_IDX,
)
from netam.codon_table import (
    CODON_AA_INDICATOR_MATRIX,
    build_stop_codon_indicator_tensor,
)


class DASMDataset(DXSMDataset):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert len(self.nt_parents) == len(self.nt_children)
        # We need to add codon index tensors to the dataset.

        self.max_codon_seq_len = self.max_aa_seq_len
        self.codon_parents_idxss = torch.full_like(
            self.aa_parents_idxss, AMBIGUOUS_CODON_IDX
        )
        self.codon_children_idxss = self.codon_parents_idxss.clone()

        # We are using the modified nt_parents and nt_children here because we
        # don't want any funky symbols in our codon indices.
        for i, (nt_parent, nt_child) in enumerate(
            zip(self.nt_parents, self.nt_children)
        ):
            assert len(nt_parent) % 3 == 0
            codon_seq_len = len(nt_parent) // 3
            self.codon_parents_idxss[i, :codon_seq_len] = codon_idx_tensor_of_str_ambig(
                nt_parent
            )
            self.codon_children_idxss[i, :codon_seq_len] = (
                codon_idx_tensor_of_str_ambig(nt_child)
            )
        assert torch.max(self.codon_parents_idxss) <= AMBIGUOUS_CODON_IDX

    def update_neutral_probs(self):
        """Update the neutral mutation probabilities for the dataset.

        This is a somewhat vague name, but that's because it includes all of the various
        types of neutral mutation probabilities that we might want to compute.

        In this case it's the neutral codon probabilities.
        """
        neutral_codon_probs_l = []

        for nt_parent, mask, nt_rates, nt_csps, branch_length in zip(
            self.nt_parents,
            self.masks,
            self.nt_ratess,
            self.nt_cspss,
            self._branch_lengths,
        ):
            # Note we are replacing all Ns with As, which means that we need to be careful
            # with masking out these positions later. We do this below.
            parent_idxs = nt_idx_tensor_of_str(nt_parent.replace("N", "A"))
            parent_len = len(nt_parent)

            mut_probs = 1.0 - torch.exp(-branch_length * nt_rates[:parent_len])
            nt_csps = nt_csps[:parent_len, :]
            nt_mask = mask.repeat_interleave(3)[: len(nt_parent)]
            molevol.check_csps(parent_idxs[nt_mask], nt_csps[: len(nt_parent)][nt_mask])

            neutral_codon_probs = molevol.neutral_codon_probs(
                parent_idxs.reshape(-1, 3),
                mut_probs.reshape(-1, 3),
                nt_csps.reshape(-1, 3, 4),
                multihit_model=self.multihit_model,
            )

            if not torch.isfinite(neutral_codon_probs).all():
                print("Found a non-finite neutral_codon_prob")
                print(f"nt_parent: {nt_parent}")
                print(f"mask: {mask}")
                print(f"nt_rates: {nt_rates}")
                print(f"nt_csps: {nt_csps}")
                print(f"branch_length: {branch_length}")
                raise ValueError(
                    f"neutral_codon_probs is not finite: {neutral_codon_probs}"
                )

            # Ensure that all values are positive before taking the log later
            neutral_codon_probs = clamp_probability(neutral_codon_probs)

            pad_len = self.max_aa_seq_len - neutral_codon_probs.shape[0]
            if pad_len > 0:
                neutral_codon_probs = F.pad(
                    neutral_codon_probs, (0, 0, 0, pad_len), value=1e-8
                )
            # Here we zero out masked positions.
            neutral_codon_probs *= mask[:, None]

            neutral_codon_probs_l.append(neutral_codon_probs)

        # Note that our masked out positions will have a nan log probability,
        # which will require us to handle them correctly downstream.
        self.log_neutral_codon_probss = torch.log(torch.stack(neutral_codon_probs_l))

    def __getitem__(self, idx):
        return {
            "codon_parents_idxs": self.codon_parents_idxss[idx],
            "codon_children_idxs": self.codon_children_idxss[idx],
            "aa_parents_idxs": self.aa_parents_idxss[idx],
            "aa_children_idxs": self.aa_children_idxss[idx],
            "subs_indicator": self.aa_subs_indicators[idx],
            "mask": self.masks[idx],
            "log_neutral_codon_probs": self.log_neutral_codon_probss[idx],
            "nt_rates": self.nt_ratess[idx],
            "nt_csps": self.nt_cspss[idx],
        }

    def move_data_to_device(self, device):
        self.codon_parents_idxss = self.codon_parents_idxss.to(device)
        self.codon_children_idxss = self.codon_children_idxss.to(device)
        self.aa_parents_idxss = self.aa_parents_idxss.to(device)
        self.aa_children_idxss = self.aa_children_idxss.to(device)
        self.aa_subs_indicators = self.aa_subs_indicators.to(device)
        self.masks = self.masks.to(device)
        self.log_neutral_codon_probss = self.log_neutral_codon_probss.to(device)
        self.nt_ratess = self.nt_ratess.to(device)
        self.nt_cspss = self.nt_cspss.to(device)
        if self.multihit_model is not None:
            self.multihit_model = self.multihit_model.to(device)


class DASMBurrito(DXSMBurrito):

    model_type = "dasm"

    @property
    def stop_codon_zapper(self):
        return self._stop_codon_zapper.to(self.device)

    @property
    def aa_codon_indicator_matrix(self):
        return CODON_AA_INDICATOR_MATRIX.T.to(self.device)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xent_loss = torch.nn.CrossEntropyLoss()
        self._stop_codon_zapper = (build_stop_codon_indicator_tensor() * -BIG).to(
            self.device
        )

    def prediction_pair_of_batch(self, batch):
        """Get log neutral codon substitution probabilities and log selection factors
        for a batch of data.

        We don't mask on the output, which will thus contain junk in all of the masked
        sites.
        """
        aa_parents_idxs = batch["aa_parents_idxs"].to(self.device)
        mask = batch["mask"].to(self.device)
        log_neutral_codon_probs = batch["log_neutral_codon_probs"].to(self.device)
        if not torch.isfinite(log_neutral_codon_probs[mask]).all():
            raise ValueError(
                f"log_neutral_codon_probs has non-finite values at relevant positions: {log_neutral_codon_probs[mask]}"
            )
        log_selection_factors = self.selection_factors_of_aa_idxs(aa_parents_idxs, mask)
        return log_neutral_codon_probs, log_selection_factors

    def predictions_of_batch(self, batch):
        """Make log probability predictions for a batch of data.

        In this case they are log probabilities of codons, which are made to be
        probabilities by setting the parent codon to 1 - sum(children).

        After all this, we clip the probabilities below to avoid log(0) issues.
        So, in cases when the sum of the children is > 1, we don't give a
        normalized probability distribution, but that won't crash the loss
        calculation because that step uses softmax.

        Note that make all ambiguous codons nan in the output, ensuring that
        they must get properly masked downstream.
        """
        log_neutral_codon_probs, log_selection_factors = self.prediction_pair_of_batch(
            batch
        )

        # This code block, in other burritos, is done in a separate function,
        # but we can't do that here because we need to normalize the
        # probabilities in a way that is not possible without having the index
        # of the parent codon. Namely, we need to set the parent codon to 1 -
        # sum(children).

        # The aa_codon_indicator_matrix lifts things up from aa land to codon land.
        log_preds = (
            log_neutral_codon_probs
            + log_selection_factors @ self.aa_codon_indicator_matrix
            + self.stop_codon_zapper
        )
        assert torch.isnan(log_preds).sum() == 0

        parent_indices = batch["codon_parents_idxs"].to(self.device)  # Shape: [B, L]
        valid_mask = parent_indices != AMBIGUOUS_CODON_IDX  # Shape: [B, L]

        # Convert to linear space so we can add probabilities.
        preds = torch.exp(log_preds)

        # Zero out the parent indices in preds, while keeping the computation
        # graph intact.
        preds_zeroer = torch.ones_like(preds)
        preds_zeroer[valid_mask, parent_indices[valid_mask]] = 0.0
        preds = preds * preds_zeroer

        # Calculate the non-parent sum after zeroing out the parent indices.
        non_parent_sum = preds[valid_mask, :].sum(dim=-1)

        # Add these parent values back in, again keeping the computation graph intact.
        preds_parent = torch.zeros_like(preds)
        preds_parent[valid_mask, parent_indices[valid_mask]] = 1.0 - non_parent_sum
        preds = preds + preds_parent

        # We have to clamp the predictions to avoid log(0) issues.
        preds = torch.clamp(preds, min=torch.finfo(preds.dtype).eps)

        log_preds = torch.log(preds)

        # Set ambiguous codons to nan to make sure that we handle them correctly downstream.
        log_preds[~valid_mask, :] = float("nan")

        return log_preds

    def loss_of_batch(self, batch):
        codon_children_idxs = batch["codon_children_idxs"].to(self.device)
        mask = batch["mask"].to(self.device)

        predictions = self.predictions_of_batch(batch)[mask]
        assert torch.isnan(predictions).sum() == 0
        codon_children_idxs = codon_children_idxs[mask]

        return self.xent_loss(predictions, codon_children_idxs)
