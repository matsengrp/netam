import torch

from netam.framework import create_mutation_and_base_indicator

# Pytest test function
def test_create_mutation_and_base_indicator():
    parent = "ACGTACTG"
    child_ = "AGGTACCG"
    site_count = 9

    expected_mutation_indi = torch.tensor([0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=torch.bool)
    expected_new_base_idxs = torch.tensor([-1, 2, -1, -1, -1, -1, 1, -1, -1], dtype=torch.int64)

    mutation_indicator, new_base_idxs = create_mutation_and_base_indicator(parent, child_, site_count)

    assert torch.equal(mutation_indicator, expected_mutation_indi), "Mutation indicators do not match."
    assert torch.equal(new_base_idxs, expected_new_base_idxs), "New base indices do not match."
