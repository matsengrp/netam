import os

import numpy as np
import torch
import pytest

from netam.framework import (
    crepe_exists,
    load_crepe,
    load_and_add_shm_model_outputs_to_pcp_df,
)
from netam.common import aa_idx_tensor_of_str_ambig, MAX_AMBIG_AA_IDX
from netam.models import TransformerBinarySelectionModelWiggleAct
from netam.dnsm import DNSMBurrito, train_test_datasets_of_pcp_df

from multiprocessing import Pool


def test_aa_idx_tensor_of_str_ambig():
    input_seq = "ACX"
    expected_output = torch.tensor([0, 1, MAX_AMBIG_AA_IDX], dtype=torch.int)
    output = aa_idx_tensor_of_str_ambig(input_seq)
    assert torch.equal(output, expected_output)


@pytest.fixture
def pcp_df():
    df = load_and_add_shm_model_outputs_to_pcp_df(
        "data/wyatt-10x-1p5m_pcp_2023-11-30_NI.first100.csv.gz",
        "data/cnn_joi_sml-shmoof_small",
    )
    return df


@pytest.fixture
def dnsm_burrito(pcp_df):
    """Fixture that returns the DNSM Burrito object."""
    train_dataset, val_dataset = train_test_datasets_of_pcp_df(pcp_df)

    model = TransformerBinarySelectionModelWiggleAct(
        nhead=2, d_model_per_head=4, dim_feedforward=256, layer_count=2
    )

    burrito = DNSMBurrito(
        train_dataset,
        val_dataset,
        model,
        batch_size=32,
        learning_rate=0.001,
        min_learning_rate=0.0001,
    )
    burrito.joint_train(epochs=1, cycle_count=2, training_method="full")
    return burrito


def test_parallel_branch_length_optimization(dnsm_burrito):
    dataset = dnsm_burrito.val_dataset
    parallel_branch_lengths = dnsm_burrito.find_optimal_branch_lengths(dataset)
    branch_lengths = dnsm_burrito.serial_find_optimal_branch_lengths(dataset)
    assert torch.allclose(branch_lengths, parallel_branch_lengths)


def test_crepe_roundtrip(dnsm_burrito):
    os.makedirs("_ignore", exist_ok=True)
    crepe_path = "_ignore/dnsm"
    dnsm_burrito.save_crepe(crepe_path)
    assert crepe_exists(crepe_path)
    crepe = load_crepe(crepe_path)
    model = crepe.model
    assert isinstance(model, TransformerBinarySelectionModelWiggleAct)
    assert dnsm_burrito.model.hyperparameters == model.hyperparameters
    model.to(dnsm_burrito.device)
    for t1, t2 in zip(
        dnsm_burrito.model.state_dict().values(), model.state_dict().values()
    ):
        assert torch.equal(t1, t2)
