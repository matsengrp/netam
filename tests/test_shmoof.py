import itertools

import pandas as pd
import pytest
import torch

import netam.framework as framework
from netam.common import BASES
from netam.framework import SHMoofDataset, Burrito
from netam.models import SHMoofModel


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
    burrito = Burrito(tiny_dataset, tiny_val_dataset, tiny_model)
    burrito.train(epochs=5)
    return burrito


def test_make_dataset(tiny_dataset):
    encoded_parent, mask, mutation_indicator = tiny_dataset[0]
    assert (mask == torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)).all()
    # First kmer is NAT due to padding, but our encoding defaults this to "N".
    assert encoded_parent[0].item() == tiny_dataset.encoder.kmer_to_index["N"]
    assert (
        mutation_indicator == torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.bool)
    ).all()


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
