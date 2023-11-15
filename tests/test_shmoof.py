import itertools

import pandas as pd
import pytest
import torch

from netam.shmoof import SHMoofDataset, SHMoofModel, SHMoofBurrito, BASES
from netam.noof import NoofModel, NoofBurrito


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
    encoded_parent, _, _ = tiny_dataset[0]
    tiny_model.forward(encoded_parent)


def test_run_shmoof(tiny_dataset, tiny_val_dataset, tiny_model):
    burrito = SHMoofBurrito(tiny_dataset, tiny_val_dataset, tiny_model)
    burrito.train(epochs=5)
    burrito.write_shmoof_output("_ignore")


def test_run_noof(tiny_dataset, tiny_val_dataset, tiny_model):
    model = NoofModel(
        tiny_dataset,
        embedding_dim=2,
        nhead=2,
        dim_feedforward=512,
        layer_count=3,
        dropout=0.5,
    )
    burrito = NoofBurrito(tiny_dataset, tiny_val_dataset, model)
    burrito.train(epochs=5)
