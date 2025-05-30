import os

import torch
import pytest

from netam.framework import (
    crepe_exists,
    load_crepe,
)
from netam.pretrained import load_multihit
from netam.common import BIG, force_spawn
from netam.models import TransformerBinarySelectionModelWiggleAct
from netam.dasm import (
    DASMBurrito,
    DASMDataset,
)
from netam.dxsm import zap_predictions_along_diagonal
from netam.sequences import (
    MAX_KNOWN_TOKEN_COUNT,
    AA_AMBIG_IDX,
    TOKEN_STR_SORTED,
    token_mask_of_aa_idxs,
)
from netam.codon_table import CODON_AA_INDICATOR_MATRIX
from conftest import get_pcp_df


@pytest.fixture(scope="module", params=["heavy", "paired"])
def dasm_burrito(request):
    pcp_df = get_pcp_df(request.param)
    force_spawn()
    """Fixture that returns the DASM Burrito object."""
    pcp_df["in_train"] = True
    pcp_df.loc[pcp_df.index[-15:], "in_train"] = False

    model = TransformerBinarySelectionModelWiggleAct(
        nhead=2,
        d_model_per_head=4,
        dim_feedforward=256,
        layer_count=2,
        output_dim=20,
        model_type="dasm",
    )

    train_dataset, val_dataset = DASMDataset.train_val_datasets_of_pcp_df(
        pcp_df,
        MAX_KNOWN_TOKEN_COUNT,
        multihit_model=load_multihit(model.multihit_model_name),
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


def test_split_recombine(dasm_burrito):
    # This is a silly test, but it helped me catch a bug resulting from
    # re-computing mask from nt strings with tokens stripped out in the split
    # method, so I'm leaving it in.
    dataset = dasm_burrito.val_dataset
    splits = dataset.split(2)
    parallel_tokens = torch.concat(
        [token_mask_of_aa_idxs(dset.aa_parents_idxss) for dset in splits]
    )
    tokens = token_mask_of_aa_idxs(dataset.aa_parents_idxss)
    assert torch.allclose(tokens, parallel_tokens)


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
    codon_predictions = dasm_burrito.predictions_of_batch(batch)
    predictions = torch.log((torch.exp(codon_predictions) @ CODON_AA_INDICATOR_MATRIX))
    aa_parents_idxs = batch["aa_parents_idxs"].to(dasm_burrito.device)
    # These sites are set to NaN, so we need to make them zero for comparison
    invalid_mask = aa_parents_idxs >= AA_AMBIG_IDX  # Shape: [B, L]
    predictions[invalid_mask] = 0.0
    zeroed_predictions = predictions.clone()
    zeroed_predictions = zap_predictions_along_diagonal(
        zeroed_predictions, aa_parents_idxs
    )
    print(predictions.shape, aa_parents_idxs.shape)
    print(predictions)
    print(aa_parents_idxs)
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
