import pandas as pd
from netam.framework import load_crepe
from netam.dnsm import TransformerBinarySelectionModel, DNSMBurrito
from epam.shmple_precompute import load_and_convert_to_tensors
import pytest


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
    dnsm = crepe.dnsm
    assert isinstance(dnsm, TransformerBinarySelectionModel)
    assert dnsm_burrito.dnsm.hyperparameters == dnsm.hyperparameters
    assert dnsm_burrito.dnsm.state_dict() == dnsm.state_dict()
    assert dnsm_burrito.encoder == crepe.encoder
