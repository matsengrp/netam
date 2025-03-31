"""Pytest tests for the simulation module."""

import pytest
import torch
import random
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from ete3 import Tree

from netam.framework import (
    load_crepe,
    codon_probs_of_parent_seq,
    trimmed_shm_model_outputs_of_crepe,
)
from netam import pretrained
from netam.common import force_spawn
from netam.dasm import (
    DASMBurrito,
    DASMDataset,
)
from netam.molevol import neutral_codon_probs_of_seq

from netam.models import TransformerBinarySelectionModelWiggleAct
from netam.sequences import (
    MAX_KNOWN_TOKEN_COUNT,
    AA_AMBIG_IDX,
    TOKEN_STR_SORTED,
    token_mask_of_aa_idxs,
    aa_mask_tensor_of,
    translate_sequence,
)
from netam.codon_table import CODON_AA_INDICATOR_MATRIX


@pytest.fixture()
def dasm_pred_burrito(pcp_df):
    pcp_df = pcp_df.head(10)
    force_spawn()
    """Fixture that returns the DASM Burrito object."""
    pcp_df["in_train"] = False
    train_dataset, val_dataset = DASMDataset.train_val_datasets_of_pcp_df(
        pcp_df, MAX_KNOWN_TOKEN_COUNT
    )
    train_dataset = val_dataset

    model = TransformerBinarySelectionModelWiggleAct(
        nhead=2,
        d_model_per_head=4,
        dim_feedforward=256,
        layer_count=2,
        output_dim=20,
    )

    burrito = DASMBurrito(
        train_dataset,
        val_dataset,
        model,
        batch_size=32,
        learning_rate=0.001,
        min_learning_rate=0.0001,
    )
    # TODO uncomment this when ready
    # burrito.joint_train(
    #     epochs=1, cycle_count=2, training_method="full", optimize_bl_first_cycle=False
    # )
    return burrito


# Test that the dasm burrito computes the same predictions as
# framework.codon_probs_of_parent_seq:


def test_neutral_probs(pcp_df, dasm_pred_burrito):
    """Test that the DASM burrito computes the same predictions as
    codon_probs_of_parent_seq."""
    pcp_df = pcp_df.head(10)
    parent_seqs = list(zip(pcp_df["parent_h"].tolist(), pcp_df["parent_l"].tolist()))

    print("recomputing branch lengths")
    dasm_pred_burrito.standardize_and_optimize_branch_lengths()
    print("updating neutral probs")
    dasm_pred_burrito.val_dataset.update_neutral_probs()
    burrito_preds = dasm_pred_burrito.val_dataset.log_neutral_codon_probss

    branch_lengths = dasm_pred_burrito.val_dataset.branch_lengths

    neutral_crepe = pretrained.load("ThriftyHumV0.2-45")
    codon_probs = []
    for (nt_parent, _), branch_length in zip(parent_seqs, branch_lengths):
        rates, csps = trimmed_shm_model_outputs_of_crepe(neutral_crepe, [nt_parent])
        mask = aa_mask_tensor_of(translate_sequence(nt_parent))
        codon_probs.append(
            neutral_codon_probs_of_seq(
                nt_parent,
                mask,
                rates[0],
                csps[0],
                branch_length,
                multihit_model=None,
            ).log()
        )
    # TODO used to be this, but this zaps the diagonal and we can't apply that as a correction to codon probs:

    # Check that the predictions match
    for pred, heavy_codon_prob in zip(burrito_preds, codon_probs):
        print(pred[0].detach().numpy())
        print(heavy_codon_prob[0].detach().numpy())
        print(pred[0].logsumexp(0))
        print(heavy_codon_prob[0].logsumexp(0))
        print((pred[0] - heavy_codon_prob[0]).detach().numpy())
        assert torch.allclose(
            pred[: len(heavy_codon_prob)], heavy_codon_prob
        ), "Predictions should match"


def test_selection_probs(pcp_df, dasm_pred_burrito):
    """Test that the DASM burrito computes the same predictions as
    codon_probs_of_parent_seq."""
    pcp_df = pcp_df.head(10)
    parent_seqs = list(zip(pcp_df["parent_h"].tolist(), pcp_df["parent_l"].tolist()))

    print("recomputing branch lengths")
    dasm_pred_burrito.standardize_and_optimize_branch_lengths()
    print("updating neutral probs")
    dasm_pred_burrito.val_dataset.update_neutral_probs()
    # Get the predictions from the DASM burrito
    dasm_pred_burrito.batch_size = 500
    val_loader = dasm_pred_burrito.build_val_loader()
    # There should be exactly one batch
    (batch,) = val_loader
    print("Getting predictions")
    burrito_preds = dasm_pred_burrito.predictions_of_batch(batch)

    branch_lengths = dasm_pred_burrito.val_dataset.branch_lengths

    # Get the predictions from codon_probs_of_parent_seq
    dasm_crepe = dasm_pred_burrito.to_crepe()
    neutral_crepe = pretrained.load("ThriftyHumV0.2-45")
    print("Computing from scratch")
    codon_probs = list(
        codon_probs_of_parent_seq(
            dasm_crepe,
            parent_seq,
            branch_length,
            neutral_crepe=neutral_crepe,
            multihit_model=None,
        ).log()
        for parent_seq, branch_length in zip(parent_seqs, branch_lengths)
    )

    # Check that the predictions match
    for pred, (heavy_codon_prob, _) in zip(burrito_preds, codon_probs):
        print(pred[0].detach().numpy())
        print(heavy_codon_prob[0].detach().numpy())
        print(pred[0].logsumexp(0))
        print(heavy_codon_prob[0].logsumexp(0))
        print((pred[0] - heavy_codon_prob[0]).detach().numpy())
        assert torch.allclose(
            pred[: len(heavy_codon_prob)], heavy_codon_prob
        ), "Predictions should match"
