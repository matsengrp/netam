import torch

from netam.framework import encode_mut_pos_and_base


def test_encode_mut_pos_and_base():
    parent = "ACGTACTG"
    child_ = "AGGTACCG"
    site_count = 9

    expected_mutation_indi = torch.tensor([0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=torch.bool)
    expected_new_base_idxs = torch.tensor(
        [-1, 2, -1, -1, -1, -1, 1, -1, -1], dtype=torch.int64
    )

    mutation_indicator, new_base_idxs = encode_mut_pos_and_base(
        parent, child_, site_count
    )

    assert torch.equal(
        mutation_indicator, expected_mutation_indi
    ), "Mutation indicators do not match."
    assert torch.equal(
        new_base_idxs, expected_new_base_idxs
    ), "New base indices do not match."
