import torch
import pandas as pd
import pytest
from netam.dasm import zap_predictions_along_diagonal, DASMBurrito, DASMDataset
from netam.common import force_spawn
from tqdm import tqdm

from netam.framework import load_crepe
from netam.sequences import set_wt_to_nan


@pytest.fixture(scope="module")
def fixed_dasm_val_burrito(pcp_df):
    force_spawn()
    """Fixture that returns the DNSM Burrito object."""
    pcp_df["in_train"] = True
    pcp_df.loc[pcp_df.index[-15:], "in_train"] = False
    dasm_crepe = load_crepe("tests/old_models/dasm_13k-v1jaffe+v1tang-joint")
    model = dasm_crepe.model
    train_dataset, val_dataset = DASMDataset.train_val_datasets_of_pcp_df(
        pcp_df, model.known_token_count
    )

    burrito = DASMBurrito(
        train_dataset,
        val_dataset,
        model,
        batch_size=32,
        learning_rate=0.001,
        min_learning_rate=0.0001,
    )
    burrito.standardize_and_optimize_branch_lengths()
    return burrito


def test_predictions_of_batch(fixed_dasm_val_burrito):
    branch_lengths = torch.tensor(
        pd.read_csv("tests/old_models/val_branch_lengths.csv")["branch_length"]
    ).double()
    these_branch_lengths = fixed_dasm_val_burrito.val_dataset.branch_lengths.double()
    assert torch.allclose(branch_lengths, these_branch_lengths)
    fixed_dasm_val_burrito.model.eval()
    val_loader = fixed_dasm_val_burrito.build_val_loader()
    predictions_list = []
    for batch in tqdm(val_loader, desc="Calculating model predictions"):
        predictions = zap_predictions_along_diagonal(
            fixed_dasm_val_burrito.predictions_of_batch(batch), batch["aa_parents_idxs"]
        )
        predictions_list.append(predictions.detach().cpu())
    these_predictions = torch.cat(predictions_list, axis=0).double()
    predictions = torch.load("tests/old_models/val_predictions.pt").double()
    assert torch.allclose(predictions, these_predictions)


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
