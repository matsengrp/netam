"""Here we define a model that outputs a vector of 20 amino acid preferences, using a
protein model embedding as input.

PIE stands for Protein Input Embedding.

Right now it's specialized to ESM2 (see below).
"""

import torch
import torch.nn.functional as F

import esm

from netam.dasm import DASMDataset, DASMBurrito
from netam.sequences import (
    translate_sequences,
)


class DASMPIEDataset(DASMDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We are specifically using ESM models here. Ideally we'd pass an
        # Embedder object and use that to do the tokenization, but we only have
        # one model here and I don't want to break the interface.
        # Note that all ESM2 models use the ESM-1b alphabet
        # https://github.com/facebookresearch/esm/blob/main/esm/pretrained.py#L175
        alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        batch_converter = alphabet.get_batch_converter()
        aa_parents = translate_sequences(self.nt_parents)
        _, _, self.pie_tokens = batch_converter(
            [(f"seq_{i}", seq) for i, seq in enumerate(aa_parents)]
        )

    def __getitem__(self, idx):
        return {
            "aa_parents_idxs": self.aa_parents_idxss[idx],
            "aa_children_idxs": self.aa_children_idxss[idx],
            "pie_tokens": self.pie_tokens[idx],
            "subs_indicator": self.aa_subs_indicators[idx],
            "mask": self.masks[idx],
            "log_neutral_aa_probs": self.log_neutral_aa_probss[idx],
            "nt_rates": self.nt_ratess[idx],
            "nt_csps": self.nt_cspss[idx],
        }

    def to(self, device):
        self.aa_parents_idxss = self.aa_parents_idxss.to(device)
        self.aa_children_idxss = self.aa_children_idxss.to(device)
        self.pie_tokens = self.pie_tokens.to(device)
        self.aa_subs_indicators = self.aa_subs_indicators.to(device)
        self.masks = self.masks.to(device)
        self.log_neutral_aa_probss = self.log_neutral_aa_probss.to(device)
        self.nt_ratess = self.nt_ratess.to(device)
        self.nt_cspss = self.nt_cspss.to(device)
        if self.multihit_model is not None:
            self.multihit_model = self.multihit_model.to(device)


class DASMPIEBurrito(DASMBurrito):

    def prediction_pair_of_batch(self, batch):
        """Get log neutral AA probabilities and log selection factors for a batch of
        data."""
        pie_tokens = batch["pie_tokens"].to(self.device)
        mask = batch["mask"].to(self.device)
        log_neutral_aa_probs = batch["log_neutral_aa_probs"].to(self.device)
        if not torch.isfinite(log_neutral_aa_probs[mask]).all():
            raise ValueError(
                f"log_neutral_aa_probs has non-finite values at relevant positions: {log_neutral_aa_probs[mask]}"
            )
        log_selection_factors = self.model(pie_tokens, mask)
        return log_neutral_aa_probs, log_selection_factors
