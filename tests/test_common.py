import torch

from netam.common import mask_tensor_of

def test_mask_tensor_of():
    input_seq = "NAAA"
    expected_output = torch.tensor([0, 1, 1, 1, 0], dtype=torch.bool)
    output = mask_tensor_of(input_seq, length=5)
    assert torch.equal(output, expected_output)