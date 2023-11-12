import pandas as pd
import pytest
import torch

from netam.shmoof import SHMoofDataset, SHMoofModel, SHMoofBurrito

@pytest.fixture
def tiny_dataset():
    df = pd.DataFrame({"parent": ["ATGTA", "GTAC"], "child": ["ACGTA", "ATACG"]})
    return SHMoofDataset(df, max_length=6, kmer_length=3)

def test_make_dataset(tiny_dataset):
    encoded_parent, mask, mutation_indicator = tiny_dataset[0]
    assert (mask == torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)).all()
    # First kmer is NAT due to padding
    assert encoded_parent[0].item() == tiny_dataset.kmer_to_index["NAT"]
    assert (mutation_indicator == torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.bool)).all()

def test_make_model(tiny_dataset):
    model = SHMoofModel(tiny_dataset)
    assert tiny_dataset.max_length == model.site_count
    encoded_parent, _, _ = tiny_dataset[0]
    model.forward(encoded_parent)

def test_run():
    train_dataframe = pd.DataFrame({"parent": ["ATGTA", "GTAC"], "child": ["ACGTA", "ATACG"]})
    val_dataframe = pd.DataFrame({"parent": ["ATGTA", "GTAA"], "child": ["ACGTA", "ATACG"]})
    burrito = SHMoofBurrito(train_dataframe, val_dataframe, max_length=300, kmer_length=5)
    burrito.train(epochs=5)
    burrito.write_shmoof_output("_ignore")