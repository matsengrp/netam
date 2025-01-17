import os

import torch
import pytest

from netam.framework import (
    crepe_exists,
    load_crepe,
)
from netam.common import aa_idx_tensor_of_str_ambig, force_spawn
from netam.models import TransformerBinarySelectionModelWiggleAct
from netam.dnsm import DNSMBurrito, DNSMDataset
from netam.sequences import AA_AMBIG_IDX, MAX_EMBEDDING_DIM, TOKEN_STR_SORTED


def test_aa_idx_tensor_of_str_ambig():
    input_seq = "ACX"
    expected_output = torch.tensor([0, 1, AA_AMBIG_IDX], dtype=torch.int)
    output = aa_idx_tensor_of_str_ambig(input_seq)
    assert torch.equal(output, expected_output)


@pytest.fixture(scope="module", params=["pcp_df", "pcp_df_paired"])
def dnsm_burrito(pcp_df):
    """Fixture that returns the DNSM Burrito object."""
    force_spawn()
    pcp_df["in_train"] = True
    pcp_df.loc[pcp_df.index[-15:], "in_train"] = False
    train_dataset, val_dataset = DNSMDataset.train_val_datasets_of_pcp_df(
        pcp_df, MAX_EMBEDDING_DIM
    )

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
    force_spawn()
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


# TODO this won't work until build_selection_matrix_from_parent is fixed
def test_build_selection_matrix_from_parent(dasm_burrito):
    dataset_row = dasm_burrito.val_dataset[0]
    
    parent = dasm_burrito.val_dataset.nt_parents[0]
    parent_aa_idxs = dasm_burrito.val_dataset.aa_parents_idxss[0]
    aa_mask = dasm_burrito.val_dataset.masks[0]
    aa_parent = "".join(TOKEN_STR_SORTED[i] for i in parent)

    separator_idx = aa_parent.index('^') * 3
    light_chain_seq = parent[:separator_idx]
    heavy_chain_seq = parent[separator_idx + 3:]
    
    direct_val = dasm_burrito.build_selection_matrix_from_parent_aa(parent_aa_idxs, aa_mask)

    indirect_val = dasm_burrito.build_selection_matrix_from_parent((light_chain_seq, heavy_chain_seq))

    assert torch.allclose(direct_val, indirect_val)
