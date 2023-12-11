import pandas as pd
import torch
import pytest

from netam.framework import load_crepe
from netam.models import TransformerBinarySelectionModel
from netam.dnsm import DNSMBurrito
from epam.shmple_precompute import load_and_convert_to_tensors


@pytest.fixture
def dnsm_burrito():
    """Fixture that returns the DNSM Burrito object."""
    pcp_df = load_and_convert_to_tensors(
        "/Users/matsen/data/wyatt-10x-1p5m_pcp_2023-10-07.first100.shmple.hdf5"
    )

    pcp_df = pcp_df[pcp_df["parent"] != pcp_df["child"]]
    print(f"After filtering out identical PCPs, we have {len(pcp_df)} PCPs.")

    dnsm = TransformerBinarySelectionModel(nhead=2, dim_feedforward=256, layer_count=2)

    burrito = DNSMBurrito(
        pcp_df,
        dnsm,
        batch_size=32,
        learning_rate=0.001,
        checkpoint_dir="./_checkpoints",
        log_dir="./_logs",
    )
    burrito.train(2)
    burrito.optimize_branch_lengths()
    return burrito


def test_crepe_roundtrip(dnsm_burrito):
    dnsm_burrito.save_crepe("_ignore/dnsm")
    crepe = load_crepe("_ignore/dnsm")
    dnsm = crepe.model
    assert isinstance(dnsm, TransformerBinarySelectionModel)
    assert dnsm_burrito.dnsm.hyperparameters == dnsm.hyperparameters
    for t1, t2 in zip(
        dnsm_burrito.dnsm.state_dict().values(), dnsm.state_dict().values()
    ):
        assert torch.equal(t1, t2)
