import torch

from netam.common import (
    nt_mask_tensor_of,
    aa_mask_tensor_of,
    codon_mask_tensor_of,
    aa_strs_from_idx_tensor,
)


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


def test_codon_mask_tensor_of():
    input_seq = "NAAAAAAAAAA"
    # First test as nucleotides.
    expected_output = torch.tensor([0, 1, 1, 0, 0], dtype=torch.bool)
    output = codon_mask_tensor_of(input_seq, aa_length=5)
    assert torch.equal(output, expected_output)
    input_seq2 = "AAAANAAAAAA"
    expected_output = torch.tensor([0, 0, 1, 0, 0], dtype=torch.bool)
    output = codon_mask_tensor_of(input_seq, input_seq2, aa_length=5)
    assert torch.equal(output, expected_output)


def test_aa_strs_from_idx_tensor():
    aa_idx_tensor = torch.tensor([[0, 1, 2, 3, 20, 21], [4, 5, 19, 21, 20, 20]])
    aa_strings = aa_strs_from_idx_tensor(aa_idx_tensor)
    assert aa_strings == ["ACDEX^", "FGY^"]
