import itertools

import pandas as pd
import pytest
import torch

from netam.shmoof import SHMoofDataset, SHMoofModel, SHMoofBurrito, BASES


@pytest.fixture
def tiny_dataset():
    df = pd.DataFrame({"parent": ["ATGTA", "GTAC"], "child": ["ACGTA", "ATAC"]})
    return SHMoofDataset(df, max_length=6, kmer_length=3)


@pytest.fixture
def tiny_model(tiny_dataset):
    return SHMoofModel(tiny_dataset, embedding_dim=1)


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


def test_run():
    train_dataframe = pd.DataFrame(
        {"parent": ["ATGTA", "GTAC"], "child": ["ACGTA", "TACG"]}
    )
    val_dataframe = pd.DataFrame(
        {"parent": ["ATGTA", "GTAA"], "child": ["ACGTA", "TACG"]}
    )
    burrito = SHMoofBurrito(train_dataframe, val_dataframe, max_length=6, kmer_length=3)
    burrito.train(epochs=5)
    burrito.write_shmoof_output("_ignore")
