import pandas as pd
import pytest
import torch

from netam.shmoof import SHMoofDataset, SHMoofModel 

@pytest.fixture
def tiny_dataset():
    df = pd.DataFrame({"parent": ["ATGTA", "GTAC"], "child": ["ACGTA", "ATACG"]})
    return SHMoofDataset(df, max_length=6, kmer_length=3)

def test_make_dataset(tiny_dataset):
    parent_seq, mask, mutation_vector = tiny_dataset[0]
    assert (mask == torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)).all()
    # First kmer is NAT due to padding
    assert parent_seq[0].item() == tiny_dataset.kmer_to_index["NAT"]
    assert (mutation_vector == torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.bool)).all()

def test_make_model(tiny_dataset):
    model = SHMoofModel(tiny_dataset)
    assert tiny_dataset.max_length == model.site_count
