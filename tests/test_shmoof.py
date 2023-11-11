import pandas as pd
import torch

from netam.shmoof import SHMoofDataset

def test_make_dataset():
    df = pd.DataFrame({"parent": ["ATGTA", "GTAC"], "child": ["ACGTA", "ATACG"]})
    dataset = SHMoofDataset(df, max_length=6, kmer_length=3)
    parent_seq, mask, mutation_vector = dataset[0]
    assert (mask == torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)).all()
    # We have one N of padding at the start, so the first kmer is NAT
    assert parent_seq[0].item() == dataset.kmer_to_index["NAT"]
    assert (mutation_vector == torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.bool)).all()