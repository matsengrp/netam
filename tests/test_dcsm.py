import torch
import pytest

from netam.common import force_spawn
from netam.sequences import MAX_KNOWN_TOKEN_COUNT
from netam.models import TransformerBinarySelectionModelWiggleAct
from netam.dcsm import (
    DCSMBurrito,
    DCSMDataset,
)


@pytest.fixture(scope="module")
def dcsm_burrito(pcp_df):
    force_spawn()
    """Fixture that returns the DNSM Burrito object."""
    pcp_df["in_train"] = True
    pcp_df.loc[pcp_df.index[-15:], "in_train"] = False
    train_dataset, val_dataset = DCSMDataset.train_val_datasets_of_pcp_df(
        pcp_df, MAX_KNOWN_TOKEN_COUNT
    )

    model = TransformerBinarySelectionModelWiggleAct(
        nhead=2,
        d_model_per_head=4,
        dim_feedforward=256,
        layer_count=2,
        output_dim=20,
    )

    burrito = DCSMBurrito(
        train_dataset,
        val_dataset,
        model,
        batch_size=32,
        learning_rate=0.001,
        min_learning_rate=0.0001,
    )
    burrito.joint_train(
        epochs=1, cycle_count=2, training_method="full", optimize_bl_first_cycle=False
    )
    return burrito


def test_parallel_branch_length_optimization(dcsm_burrito):
    dataset = dcsm_burrito.val_dataset
    parallel_branch_lengths = dcsm_burrito.find_optimal_branch_lengths(dataset)
    branch_lengths = dcsm_burrito.serial_find_optimal_branch_lengths(dataset)
    assert torch.allclose(branch_lengths, parallel_branch_lengths)
