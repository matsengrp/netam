import torch

from netam.framework import load_crepe
from netam.sequences import set_wt_to_nan


# The outputs used for this test are produced by running
# `test_backward_compat_copy.py` on the wd-old-model-runner branch.
# This is to ensure that we can still load older crepes, even if we change the
# dimensions of model layers, as we did with the Embedding layer in
# https://github.com/matsengrp/netam/pull/92.
def test_old_crepe_outputs():
    example_seq = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSSGMHWVRQAPGKGLEWVAVIWYDGSNKYYADSVKGRFTISRDNSKNTVYLQMNSLRAEDTAVYYCAREGHSNYPYYYYYMDVWGKGTTVTVSS"
    dasm_crepe = load_crepe("tests/old_models/dasm_13k-v1jaffe+v1tang-joint")
    dnsm_crepe = load_crepe("tests/old_models/dnsm_13k-v1jaffe+v1tang-joint")

    dasm_vals = torch.nan_to_num(
        set_wt_to_nan(
            torch.load("tests/old_models/dasm_output", weights_only=True), example_seq
        ),
        0.0,
    )
    dnsm_vals = torch.load("tests/old_models/dnsm_output", weights_only=True)

    dasm_result = torch.nan_to_num(dasm_crepe([example_seq])[0], 0.0)
    dnsm_result = dnsm_crepe([example_seq])[0]
    assert torch.allclose(dasm_result, dasm_vals)
    assert torch.allclose(dnsm_result, dnsm_vals)
