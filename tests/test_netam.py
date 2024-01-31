import itertools

import pandas as pd
import pytest
import torch

import netam.framework as framework
from netam.common import BIG
from netam.framework import SHMoofDataset, SHMBurrito, RSSHMBurrito
from netam.models import SHMoofModel, RSCNNModel


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
    burrito.train(epochs=5)
    return burrito


def test_make_dataset(tiny_dataset):
    encoded_parent, mask, mutation_indicator, new_base_idxs, wt_base_multiplier = tiny_dataset[0]
    assert (mask == torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)).all()
    # First kmer is NAT due to padding, but our encoding defaults this to "N".
    assert encoded_parent[0].item() == tiny_dataset.encoder.kmer_to_index["N"]
    assert (
        mutation_indicator == torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.bool)
    ).all()
    assert (
        new_base_idxs == torch.tensor([-1, 1, -1, -1, -1, -1], dtype=torch.int64)
    ).all()
    correct_wt_base_multiplier = torch.full((6, 4), 1.0)
    correct_wt_base_multiplier[1, 3] = -BIG
    assert (wt_base_multiplier == correct_wt_base_multiplier).all()


def test_run_model_forward(tiny_dataset, tiny_model):
    assert tiny_dataset.encoder.site_count == tiny_model.site_count
    tiny_model.forward(tiny_dataset.encoded_parents, tiny_dataset.masks)


def test_write_output(tiny_burrito):
    tiny_burrito.model.write_shmoof_output("_ignore")


def test_crepe_roundtrip(tiny_burrito):
    tiny_burrito.save_crepe("_ignore/tiny_crepe")
    crepe = framework.load_crepe("_ignore/tiny_crepe")
    assert crepe.encoder.parameters["site_count"] == tiny_burrito.model.site_count
    assert crepe.encoder.parameters["kmer_length"] == tiny_burrito.model.kmer_length
    assert torch.isclose(crepe.model.kmer_rates, tiny_burrito.model.kmer_rates).all()
    assert torch.isclose(crepe.model.site_rates, tiny_burrito.model.site_rates).all()
    ## Assert that crepe.model is in eval mode
    assert not crepe.model.training


@pytest.fixture
def tiny_rsmodel():
    return RSCNNModel(kmer_length=3, embedding_dim=2, filter_count=2, kernel_size=3)


@pytest.fixture
def tiny_rsburrito(tiny_dataset, tiny_val_dataset, tiny_rsmodel):
    burrito = RSSHMBurrito(tiny_dataset, tiny_val_dataset, tiny_rsmodel)
    burrito.train(epochs=5)
    return burrito


def test_write_output(tiny_rsburrito):
    tiny_rsburrito.save_crepe("_ignore/tiny_rscrepe")
