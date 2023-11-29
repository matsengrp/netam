import itertools

import pandas as pd
import pytest
import torch

from netam.framework import SHMoofDataset, Burrito, BASES
from netam.models import SHMoofModel


@pytest.fixture
def tiny_dataset():
    df = pd.DataFrame({"parent": ["ATGTA", "GTAC"], "child": ["ACGTA", "ATAC"]})
    return SHMoofDataset(df, max_length=6, kmer_length=3)


@pytest.fixture
def tiny_val_dataset():
    df = pd.DataFrame({"parent": ["ATGTA", "GTAA"], "child": ["ACGTA", "TACG"]})
    return SHMoofDataset(df, max_length=6, kmer_length=3)


@pytest.fixture
def tiny_model(tiny_dataset):
    return SHMoofModel(tiny_dataset)


def test_make_dataset(tiny_dataset):
    encoded_parent, mask, mutation_indicator = tiny_dataset[0]
    assert (mask == torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)).all()
    # First kmer is NAT due to padding, but our encoding defaults this to "N".
    assert encoded_parent[0].item() == tiny_dataset.kmer_to_index["N"]
    assert (
        mutation_indicator == torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.bool)
    ).all()


def test_run_model_forward(tiny_dataset, tiny_model):
    assert tiny_dataset.max_length == tiny_model.site_count
    tiny_model.forward(tiny_dataset.encoded_parents, tiny_dataset.masks)


def test_run_shmoof(tiny_dataset, tiny_val_dataset, tiny_model):
    burrito = Burrito(tiny_dataset, tiny_val_dataset, tiny_model)
    burrito.train(epochs=5)
    tiny_model.write_shmoof_output("_ignore")
