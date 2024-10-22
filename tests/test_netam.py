import os

import pandas as pd
import pytest
import numpy as np
import torch

import netam.framework as framework
from netam.common import BIG
from netam.framework import SHMoofDataset, SHMBurrito, RSSHMBurrito
from netam.models import SHMoofModel, RSSHMoofModel, IndepRSCNNModel


@pytest.fixture
def tiny_dataset():
    df = pd.DataFrame({"parent": ["ATGTA", "GTAC"], "child": ["ACGTA", "ATAC"]})
    return SHMoofDataset(df, site_count=6, kmer_length=3)


@pytest.fixture
def tiny_val_dataset():
    df = pd.DataFrame({"parent": ["ATGTA", "GTAA"], "child": ["ACGTA", "TACG"]})
    return SHMoofDataset(df, site_count=6, kmer_length=3)


@pytest.fixture
def tiny_model():
    return SHMoofModel(site_count=6, kmer_length=3)


@pytest.fixture
def tiny_burrito(tiny_dataset, tiny_val_dataset, tiny_model):
    burrito = SHMBurrito(tiny_dataset, tiny_val_dataset, tiny_model)
    burrito.simple_train(epochs=5)
    return burrito


def test_make_dataset(tiny_dataset):
    (
        encoded_parent,
        mask,
        mutation_indicator,
        new_base_idxs,
        wt_base_modifier,
        branch_length,
    ) = tiny_dataset[0]
    assert (mask == torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)).all()
    # First kmer is NAT due to padding, but our encoding defaults this to "N".
    assert encoded_parent[0].item() == tiny_dataset.encoder.kmer_to_index["N"]
    assert (
        mutation_indicator == torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.bool)
    ).all()
    assert (
        new_base_idxs == torch.tensor([-1, 1, -1, -1, -1, -1], dtype=torch.int64)
    ).all()
    correct_wt_base_modifier = torch.zeros((6, 4))
    correct_wt_base_modifier[0, 0] = -BIG
    correct_wt_base_modifier[1, 3] = -BIG
    correct_wt_base_modifier[2, 2] = -BIG
    correct_wt_base_modifier[3, 3] = -BIG
    correct_wt_base_modifier[4, 0] = -BIG
    assert (wt_base_modifier == correct_wt_base_modifier).all()
    assert branch_length == 1 / 5


def test_write_output(tiny_burrito):
    os.makedirs("_ignore", exist_ok=True)
    tiny_burrito.model.write_shmoof_output("_ignore")


def test_crepe_roundtrip(tiny_burrito):
    os.makedirs("_ignore", exist_ok=True)
    tiny_burrito.save_crepe("_ignore/tiny_crepe")
    crepe = framework.load_crepe("_ignore/tiny_crepe")
    assert crepe.encoder.parameters["site_count"] == tiny_burrito.model.site_count
    assert crepe.encoder.parameters["kmer_length"] == tiny_burrito.model.kmer_length
    assert torch.isclose(crepe.model.kmer_rates, tiny_burrito.model.kmer_rates).all()
    assert torch.isclose(crepe.model.site_rates, tiny_burrito.model.site_rates).all()
    ## Assert that crepe.model is in eval mode
    assert not crepe.model.training


@pytest.fixture
def tiny_rsshmoofmodel():
    return RSSHMoofModel(site_count=6, kmer_length=3)


@pytest.fixture
def tiny_rsscnnmodel():
    return IndepRSCNNModel(
        kmer_length=3, embedding_dim=2, filter_count=2, kernel_size=3
    )


@pytest.fixture
def mini_dataset():
    df = pd.read_csv("data/wyatt-10x-1p5m_pcp_2023-11-30_NI.first100.csv.gz")
    return SHMoofDataset(df, site_count=500, kmer_length=3)


@pytest.fixture
def mini_rsburrito(mini_dataset, tiny_rsscnnmodel):
    burrito = RSSHMBurrito(mini_dataset, mini_dataset, tiny_rsscnnmodel)
    burrito.joint_train(epochs=5, training_method="yun")
    return burrito


def test_write_output(mini_rsburrito):
    os.makedirs("_ignore", exist_ok=True)
    mini_rsburrito.save_crepe("_ignore/mini_rscrepe")


def test_standardize_model_rates(mini_rsburrito):
    mini_rsburrito.standardize_model_rates()
    vrc01_rate_14 = mini_rsburrito.vrc01_site_14_model_rate()
    assert np.isclose(vrc01_rate_14, 1.0)
