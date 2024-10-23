import os

import torch
import pytest

from netam.common import BIG, force_spawn
from netam.framework import (
    crepe_exists,
    load_crepe,
)
from netam.models import TransformerBinarySelectionModelWiggleAct
from netam.dasm import (
    DASMBurrito,
    DASMDataset,
    zap_predictions_along_diagonal,
)


@pytest.fixture(scope="module")
def dasm_burrito(pcp_df):
    force_spawn()
    """Fixture that returns the DNSM Burrito object."""
    pcp_df["in_train"] = True
    pcp_df.loc[pcp_df.index[-15:], "in_train"] = False
    train_dataset, val_dataset = DASMDataset.train_val_datasets_of_pcp_df(pcp_df)

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
    burrito.joint_train(
        epochs=1, cycle_count=2, training_method="full", optimize_bl_first_cycle=False
    )
    return burrito


def test_parallel_branch_length_optimization(dasm_burrito):
    dataset = dasm_burrito.val_dataset
    parallel_branch_lengths = dasm_burrito.find_optimal_branch_lengths(dataset)
    branch_lengths = dasm_burrito.serial_find_optimal_branch_lengths(dataset)
    assert torch.allclose(branch_lengths, parallel_branch_lengths)


def test_crepe_roundtrip(dasm_burrito):
    os.makedirs("_ignore", exist_ok=True)
    crepe_path = "_ignore/dasm"
    dasm_burrito.save_crepe(crepe_path)
    assert crepe_exists(crepe_path)
    crepe = load_crepe(crepe_path)
    model = crepe.model
    assert isinstance(model, TransformerBinarySelectionModelWiggleAct)
    assert dasm_burrito.model.hyperparameters == model.hyperparameters
    model.to(dasm_burrito.device)
    for t1, t2 in zip(
        dasm_burrito.model.state_dict().values(), model.state_dict().values()
    ):
        assert torch.equal(t1, t2)


def test_zap_diagonal(dasm_burrito):
    batch = dasm_burrito.val_dataset[0:2]
    predictions = dasm_burrito.predictions_of_batch(batch)
    predictions = torch.cat(
        [predictions, torch.zeros_like(predictions[:, :, :1])], dim=-1
    )
    aa_parents_idxs = batch["aa_parents_idxs"].to(dasm_burrito.device)
    zeroed_predictions = predictions.clone()
    zeroed_predictions = zap_predictions_along_diagonal(
        zeroed_predictions, aa_parents_idxs
    )
    L = predictions.shape[1]
    for batch_idx in range(2):
        for i in range(L):
            for j in range(20):
                if j == aa_parents_idxs[batch_idx, i]:
                    assert zeroed_predictions[batch_idx, i, j] == -BIG
                else:
                    assert (
                        zeroed_predictions[batch_idx, i, j]
                        == predictions[batch_idx, i, j]
                    )
