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
from netam.sequences import MAX_KNOWN_TOKEN_COUNT, TOKEN_STR_SORTED


# TODO verify that this loops through both pcp_dfs, even though one is named
# the same as the argument. If not, remember to fix in test_dnsm.py too.
@pytest.fixture(scope="module", params=["pcp_df", "pcp_df_paired"])
def dasm_burrito(pcp_df):
    force_spawn()
    """Fixture that returns the DNSM Burrito object."""
    pcp_df["in_train"] = True
    pcp_df.loc[pcp_df.index[-15:], "in_train"] = False
    train_dataset, val_dataset = DASMDataset.train_val_datasets_of_pcp_df(
        pcp_df, MAX_KNOWN_TOKEN_COUNT
    )

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


def test_selection_factors_of_aa_str(dasm_burrito):
    parent_aa_idxs = dasm_burrito.val_dataset.aa_parents_idxss[0]
    aa_parent = "".join(TOKEN_STR_SORTED[i] for i in parent_aa_idxs)
    # This won't work if we start testing with ambiguous sequences
    aa_parent = aa_parent.replace("X", "")
    aa_parent_pair = tuple(aa_parent.split("^"))
    res = dasm_burrito.model.selection_factors_of_aa_str(aa_parent_pair)
    assert len(res[0]) == len(aa_parent_pair[0])
    assert len(res[1]) == len(aa_parent_pair[1])
    assert res[0].shape[1] == 20
    assert res[1].shape[1] == 20


def test_build_selection_matrix_from_parent(dasm_burrito):
    parent = dasm_burrito.val_dataset.nt_parents[0]
    parent_aa_idxs = dasm_burrito.val_dataset.aa_parents_idxss[0]
    aa_mask = dasm_burrito.val_dataset.masks[0]
    aa_parent = "".join(TOKEN_STR_SORTED[i] for i in parent_aa_idxs)
    # This won't work if we start testing with ambiguous sequences
    aa_parent = aa_parent.replace("X", "")

    separator_idx = aa_parent.index("^") * 3
    light_chain_seq = parent[:separator_idx]
    heavy_chain_seq = parent[separator_idx + 3 :]

    direct_val = dasm_burrito.build_selection_matrix_from_parent_aa(
        parent_aa_idxs, aa_mask
    )

    indirect_val = dasm_burrito.build_selection_matrix_from_parent(
        (light_chain_seq, heavy_chain_seq)
    )

    assert torch.allclose(direct_val[: len(indirect_val[0])], indirect_val[0])
    assert torch.allclose(
        direct_val[len(indirect_val[0]) + 1 :][: len(indirect_val[1])], indirect_val[1]
    )
