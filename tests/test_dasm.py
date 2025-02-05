import os

import torch
import pytest

from netam.common import BIG, force_spawn
from netam.framework import (
    crepe_exists,
    load_crepe,
)
from netam.models import TransformerBinarySelectionModelWiggleAct
from netam.ddsm import (
    DDSMBurrito,
    DDSMDataset,
    zap_predictions_along_diagonal,
)
from netam.sequences import (
    MAX_KNOWN_TOKEN_COUNT,
    TOKEN_STR_SORTED,
    token_mask_of_aa_idxs,
)

torch.set_printoptions(precision=10)


@pytest.fixture(scope="module", params=["pcp_df", "pcp_df_paired"])
def ddsm_burrito(pcp_df):
    force_spawn()
    """Fixture that returns the DNSM Burrito object."""
    pcp_df["in_train"] = True
    pcp_df.loc[pcp_df.index[-15:], "in_train"] = False
    train_dataset, val_dataset = DDSMDataset.train_val_datasets_of_pcp_df(
        pcp_df, MAX_KNOWN_TOKEN_COUNT
    )

    model = TransformerBinarySelectionModelWiggleAct(
        nhead=2,
        d_model_per_head=4,
        dim_feedforward=256,
        layer_count=2,
        output_dim=20,
    )

    burrito = DDSMBurrito(
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


def test_parallel_branch_length_optimization(ddsm_burrito):
    dataset = ddsm_burrito.val_dataset
    parallel_branch_lengths = ddsm_burrito.find_optimal_branch_lengths(dataset)
    branch_lengths = ddsm_burrito.serial_find_optimal_branch_lengths(dataset)
    assert torch.allclose(branch_lengths, parallel_branch_lengths)


def test_split_recombine(ddsm_burrito):
    # This is a silly test, but it helped me catch a bug resulting from
    # re-computing mask from nt strings with tokens stripped out in the split
    # method, so I'm leaving it in.
    dataset = ddsm_burrito.val_dataset
    splits = dataset.split(2)
    parallel_tokens = torch.concat(
        [token_mask_of_aa_idxs(dset.aa_parents_idxss) for dset in splits]
    )
    tokens = token_mask_of_aa_idxs(dataset.aa_parents_idxss)
    assert torch.allclose(tokens, parallel_tokens)


def test_crepe_roundtrip(ddsm_burrito):
    os.makedirs("_ignore", exist_ok=True)
    crepe_path = "_ignore/ddsm"
    ddsm_burrito.save_crepe(crepe_path)
    assert crepe_exists(crepe_path)
    crepe = load_crepe(crepe_path)
    model = crepe.model
    assert isinstance(model, TransformerBinarySelectionModelWiggleAct)
    assert ddsm_burrito.model.hyperparameters == model.hyperparameters
    model.to(ddsm_burrito.device)
    for t1, t2 in zip(
        ddsm_burrito.model.state_dict().values(), model.state_dict().values()
    ):
        assert torch.equal(t1, t2)


def test_zap_diagonal(ddsm_burrito):
    batch = ddsm_burrito.val_dataset[0:2]
    predictions = ddsm_burrito.predictions_of_batch(batch)
    predictions = torch.cat(
        [predictions, torch.zeros_like(predictions[:, :, :1])], dim=-1
    )
    aa_parents_idxs = batch["aa_parents_idxs"].to(ddsm_burrito.device)
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


def test_selection_factors_of_aa_str(ddsm_burrito):
    parent_aa_idxs = ddsm_burrito.val_dataset.aa_parents_idxss[0]
    aa_parent = "".join(TOKEN_STR_SORTED[i] for i in parent_aa_idxs)
    # This won't work if we start testing with ambiguous sequences
    aa_parent = aa_parent.replace("X", "")
    aa_parent_pair = tuple(aa_parent.split("^"))
    res = ddsm_burrito.model.selection_factors_of_aa_str(aa_parent_pair)
    assert len(res[0]) == len(aa_parent_pair[0])
    assert len(res[1]) == len(aa_parent_pair[1])
    assert res[0].shape[1] == 20
    assert res[1].shape[1] == 20


def test_build_selection_matrix_from_parent(ddsm_burrito):
    parent = ddsm_burrito.val_dataset.nt_parents[0]
    parent_aa_idxs = ddsm_burrito.val_dataset.aa_parents_idxss[0]
    aa_mask = ddsm_burrito.val_dataset.masks[0]
    aa_parent = "".join(TOKEN_STR_SORTED[i] for i in parent_aa_idxs)
    # This won't work if we start testing with ambiguous sequences
    aa_parent = aa_parent.replace("X", "")

    separator_idx = aa_parent.index("^") * 3
    light_chain_seq = parent[:separator_idx]
    heavy_chain_seq = parent[separator_idx + 3 :]

    direct_val = ddsm_burrito.build_selection_matrix_from_parent_aa(
        parent_aa_idxs, aa_mask
    )

    indirect_val = ddsm_burrito._build_selection_matrix_from_parent(
        (light_chain_seq, heavy_chain_seq)
    )

    assert torch.allclose(direct_val[: len(indirect_val[0])], indirect_val[0])
    assert torch.allclose(
        direct_val[len(indirect_val[0]) + 1 :][: len(indirect_val[1])], indirect_val[1]
    )
