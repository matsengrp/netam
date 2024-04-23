import torch

from netam.common import nt_mask_tensor_of, aa_mask_tensor_of


def test_mask_tensor_of():
    input_seq = "NAAA"
    # First test as nucleotides.
    expected_output = torch.tensor([0, 1, 1, 1, 0], dtype=torch.bool)
    output = nt_mask_tensor_of(input_seq, length=5)
    assert torch.equal(output, expected_output)
    # Next test as amino acids, where N counts as an AA.
    expected_output = torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)
    output = aa_mask_tensor_of(input_seq, length=5)
    assert torch.equal(output, expected_output)
