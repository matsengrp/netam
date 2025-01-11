"""Defining the deep natural selection model (DNSM)."""

import copy

import pandas as pd
import torch
import torch.nn.functional as F

from netam.common import (
    assert_pcp_valid,
    clamp_probability,
    codon_mask_tensor_of,
    BIG,
)
from netam.dxsm import DXSMDataset, DXSMBurrito
import netam.molevol as molevol

from netam.common import aa_idx_tensor_of_str_ambig
from netam.sequences import (
    aa_idx_array_of_str,
    aa_subs_indicator_tensor_of,
    build_stop_codon_indicator_tensor,
    nt_idx_tensor_of_str,
    token_mask_of_aa_idxs,
    translate_sequence,
    translate_sequences,
    codon_idx_tensor_of_str_ambig,
    AA_AMBIG_IDX,
    AMBIGUOUS_CODON_IDX,
    CODON_AA_INDICATOR_MATRIX,
    RESERVED_TOKEN_REGEX,
    MAX_AA_TOKEN_IDX,
)


class DCSMDataset(DXSMDataset):

    def __init__(
        self,
        nt_parents: pd.Series,
        nt_children: pd.Series,
        nt_ratess: torch.Tensor,
        nt_cspss: torch.Tensor,
        branch_lengths: torch.Tensor,
        multihit_model=None,
    ):
        self.nt_parents = nt_parents.str.replace(RESERVED_TOKEN_REGEX, "N", regex=True)
        # We will replace reserved tokens with Ns but use the unmodified
        # originals for codons and mask creation.
        self.nt_children = nt_children.str.replace(
            RESERVED_TOKEN_REGEX, "N", regex=True
        )
        self.nt_ratess = nt_ratess
        self.nt_cspss = nt_cspss
        self.multihit_model = copy.deepcopy(multihit_model)
        if multihit_model is not None:
            # We want these parameters to act like fixed data. This is essential
            # for multithreaded branch length optimization to work.
            self.multihit_model.values.requires_grad_(False)

        assert len(self.nt_parents) == len(self.nt_children)
        pcp_count = len(self.nt_parents)

        # Important to use the unmodified versions of nt_parents and
        # nt_children so they still contain special tokens.
        aa_parents = translate_sequences(nt_parents)
        aa_children = translate_sequences(nt_children)

        self.max_codon_seq_len = max(len(seq) for seq in aa_parents)
        # We have sequences of varying length, so we start with all tensors set
        # to the ambiguous amino acid, and then will fill in the actual values
        # below.
        self.codon_parents_idxss = torch.full(
            (pcp_count, self.max_codon_seq_len), AMBIGUOUS_CODON_IDX
        )
        self.codon_children_idxss = self.codon_parents_idxss.clone()
        self.aa_parents_idxss = torch.full(
            (pcp_count, self.max_codon_seq_len), AA_AMBIG_IDX
        )
        self.aa_children_idxss = torch.full(
            (pcp_count, self.max_codon_seq_len), AA_AMBIG_IDX
        )
        # TODO here we are computing the subs indicators. This is handy for OE plots.
        self.aa_subs_indicators = torch.zeros((pcp_count, self.max_codon_seq_len))

        self.masks = torch.ones((pcp_count, self.max_codon_seq_len), dtype=torch.bool)

        # We are using the modified nt_parents and nt_children here because we
        # don't want any funky symbols in our codon indices.
        for i, (nt_parent, nt_child, aa_parent, aa_child) in enumerate(
            zip(self.nt_parents, self.nt_children, aa_parents, aa_children)
        ):
            self.masks[i, :] = codon_mask_tensor_of(
                nt_parent, nt_child, aa_length=self.max_codon_seq_len
            )
            assert len(nt_parent) % 3 == 0
            codon_seq_len = len(nt_parent) // 3

            assert_pcp_valid(nt_parent, nt_child, aa_mask=self.masks[i][:codon_seq_len])

            self.codon_parents_idxss[i, :codon_seq_len] = codon_idx_tensor_of_str_ambig(
                nt_parent
            )
            self.codon_children_idxss[i, :codon_seq_len] = (
                codon_idx_tensor_of_str_ambig(nt_child)
            )
            self.aa_parents_idxss[i, :codon_seq_len] = aa_idx_tensor_of_str_ambig(
                aa_parent
            )
            self.aa_children_idxss[i, :codon_seq_len] = aa_idx_tensor_of_str_ambig(
                aa_child
            )
            self.aa_subs_indicators[i, :codon_seq_len] = aa_subs_indicator_tensor_of(
                aa_parent, aa_child
            )

        assert torch.all(self.masks.sum(dim=1) > 0)
        assert torch.max(self.aa_parents_idxss) <= MAX_AA_TOKEN_IDX
        assert torch.max(self.aa_children_idxss) <= MAX_AA_TOKEN_IDX
        assert torch.max(self.codon_parents_idxss) <= AMBIGUOUS_CODON_IDX

        self._branch_lengths = branch_lengths
        self.update_neutral_probs()

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
            mask = mask.to("cpu")
            nt_rates = nt_rates.to("cpu")
            nt_csps = nt_csps.to("cpu")
            if self.multihit_model is not None:
                multihit_model = copy.deepcopy(self.multihit_model).to("cpu")
            else:
                multihit_model = None
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
                multihit_model=multihit_model,
            )

            if not torch.isfinite(neutral_codon_probs).all():
                print(f"Found a non-finite neutral_codon_prob")
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

            pad_len = self.max_codon_seq_len - neutral_codon_probs.shape[0]
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

    def to(self, device):
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


class DCSMBurrito(DXSMBurrito):

    model_type = "dcsm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xent_loss = torch.nn.CrossEntropyLoss()
        self.stop_codon_zapper = build_stop_codon_indicator_tensor() * -BIG

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
        # We need the model to see special tokens here. For every other purpose
        # they are masked out.
        keep_token_mask = mask | token_mask_of_aa_idxs(aa_parents_idxs)
        log_selection_factors = self.model(aa_parents_idxs, keep_token_mask)
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

        # This indicator lifts things up from aa land to codon land.
        # TODO I guess we could store indicator in self and have everything move with a self.to(device) call.
        indicator = CODON_AA_INDICATOR_MATRIX.to(self.device).T
        log_preds = (
            log_neutral_codon_probs
            + log_selection_factors @ indicator
            + self.stop_codon_zapper.to(self.device)
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

    # TODO copied from dasm.py
    def build_selection_matrix_from_parent(self, parent: str):
        """Build a selection matrix from a parent amino acid sequence.

        Values at ambiguous sites are meaningless.
        """
        # This is simpler than the equivalent in dnsm.py because we get the selection
        # matrix directly. Note that selection_factors_of_aa_str does the exponentiation
        # so this indeed gives us the selection factors, not the log selection factors.
        parent = translate_sequence(parent)
        per_aa_selection_factors = self.model.selection_factors_of_aa_str(parent)

        parent = parent.replace("X", "A")
        parent_idxs = aa_idx_array_of_str(parent)
        per_aa_selection_factors[torch.arange(len(parent_idxs)), parent_idxs] = 1.0

        return per_aa_selection_factors
