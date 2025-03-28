"""Pytest tests for the simulation module."""

import pytest
import torch
import random
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from ete3 import Tree

from netam.framework import load_crepe, codon_probs_of_parent_seq
from netam import pretrained
from netam.common import force_spawn
from netam.dasm import (
    DASMBurrito,
    DASMDataset,
)

from netam.models import TransformerBinarySelectionModelWiggleAct
from netam.sequences import (
    MAX_KNOWN_TOKEN_COUNT,
    AA_AMBIG_IDX,
    TOKEN_STR_SORTED,
    token_mask_of_aa_idxs,
)
from netam.codon_table import CODON_AA_INDICATOR_MATRIX


@pytest.fixture()
def dasm_pred_burrito(pcp_df):
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


def test_dasm_predictions(pcp_df, dasm_pred_burrito):
    """Test that the DASM burrito computes the same predictions as
    codon_probs_of_parent_seq."""
    parent_seqs = list(zip(pcp_df["parent_h"].tolist(), pcp_df["parent_l"].tolist()))

    # Get the predictions from the DASM burrito
    dasm_pred_burrito.batch_size = 500
    val_loader = dasm_pred_burrito.build_val_loader()
    # There should be exactly one batch
    (batch,) = val_loader
    burrito_preds = dasm_pred_burrito.predictions_of_batch(batch)

    branch_lengths = dasm_pred_burrito.val_dataset.branch_lengths

    # Get the predictions from codon_probs_of_parent_seq
    dasm_crepe = dasm_pred_burrito.to_crepe()
    neutral_crepe = pretrained.load("ThriftyHumV0.2-45")
    codon_probs = list(
        codon_probs_of_parent_seq(
            dasm_crepe,
            parent_seq,
            branch_length,
            neutral_crepe=neutral_crepe,
            multihit_model=None,
        )
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
