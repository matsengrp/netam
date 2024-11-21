import torch

from netam.common import nt_mask_tensor_of, aa_mask_tensor_of, codon_mask_tensor_of


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
