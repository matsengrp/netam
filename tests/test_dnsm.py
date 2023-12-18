import pandas as pd
import torch
import pytest

from netam.framework import load_crepe
from netam.common import aa_idx_tensor_of_str_ambig, MAX_AMBIG_AA_IDX
from netam.models import TransformerBinarySelectionModel
from netam.dnsm import DNSMBurrito, train_test_datasets_of_pcp_df
from epam.shmple_precompute import load_and_convert_to_tensors


def test_mask_tensor_of():
    input_seq = "ACX"
    expected_output = torch.tensor([0, 1, MAX_AMBIG_AA_IDX], dtype=torch.int)
    output = aa_idx_tensor_of_str_ambig(input_seq)
    assert torch.equal(output, expected_output)


@pytest.fixture
def dnsm_burrito():
    """Fixture that returns the DNSM Burrito object."""
    pcp_df = load_and_convert_to_tensors(
        "/Users/matsen/data/wyatt-10x-1p5m_pcp_2023-10-07.first100.shmple.hdf5"
    )

    pcp_df = pcp_df[pcp_df["parent"] != pcp_df["child"]]
    print(f"After filtering out identical PCPs, we have {len(pcp_df)} PCPs.")
    train_dataset, val_dataset = train_test_datasets_of_pcp_df(pcp_df)

    model = TransformerBinarySelectionModel(
        nhead=2, d_model_per_head=4, dim_feedforward=256, layer_count=2
    )

    burrito = DNSMBurrito(
        train_dataset,
        val_dataset,
        model,
        batch_size=32,
        learning_rate=0.001,
        min_learning_rate=0.0001,
        device="cpu",
    )
    burrito.train(2)
    burrito.optimize_branch_lengths()
    return burrito


def test_crepe_roundtrip(dnsm_burrito):
    dnsm_burrito.save_crepe("_ignore/dnsm")
    crepe = load_crepe("_ignore/dnsm")
    model = crepe.model
    assert isinstance(model, TransformerBinarySelectionModel)
    assert dnsm_burrito.model.hyperparameters == model.hyperparameters
    model.to(dnsm_burrito.device)
    for t1, t2 in zip(
        dnsm_burrito.model.state_dict().values(), model.state_dict().values()
    ):
        assert torch.equal(t1, t2)
