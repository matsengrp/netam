import torch
import pandas as pd
import pytest
from netam.ddsm import zap_predictions_along_diagonal, DDSMBurrito, DDSMDataset
from netam.common import force_spawn
from tqdm import tqdm

from netam.framework import load_crepe
from netam.sequences import aa_idx_tensor_of_str_ambig


@pytest.fixture(scope="module")
def fixed_ddsm_val_burrito(pcp_df):
    force_spawn()
    """Fixture that returns the DNSM Burrito object."""
    pcp_df["in_train"] = True
    pcp_df.loc[pcp_df.index[-15:], "in_train"] = False
    ddsm_crepe = load_crepe("tests/old_models/ddsm_13k-v1jaffe+v1tang-joint")
    model = ddsm_crepe.model
    train_dataset, val_dataset = DDSMDataset.train_val_datasets_of_pcp_df(
        pcp_df, model.known_token_count
    )

    burrito = DDSMBurrito(
        train_dataset,
        val_dataset,
        model,
        batch_size=32,
        learning_rate=0.001,
        min_learning_rate=0.0001,
    )
    burrito.standardize_and_optimize_branch_lengths()
    return burrito


def test_predictions_of_batch(fixed_ddsm_val_burrito):
    # These outputs were produced by the comparison code in this test, but
    # written to the files referenced here. The code state was netam 3c632fa.
    # (however, this test did not exist in the codebase at that time)
    branch_lengths = torch.tensor(
        pd.read_csv("tests/old_models/val_branch_lengths.csv")["branch_length"]
    ).double()
    these_branch_lengths = fixed_ddsm_val_burrito.val_dataset.branch_lengths.double()
    assert torch.allclose(branch_lengths, these_branch_lengths)
    fixed_ddsm_val_burrito.model.eval()
    val_loader = fixed_ddsm_val_burrito.build_val_loader()
    predictions_list = []
    for batch in tqdm(val_loader, desc="Calculating model predictions"):
        predictions = zap_predictions_along_diagonal(
            fixed_ddsm_val_burrito.predictions_of_batch(batch), batch["aa_parents_idxs"]
        )
        predictions_list.append(predictions.detach().cpu())
    these_predictions = torch.cat(predictions_list, axis=0).double()
    predictions = torch.load(
        "tests/old_models/val_predictions.pt", weights_only=True
    ).double()
    assert torch.allclose(predictions, these_predictions)


# The outputs used for this test are produced by running
# `test_backward_compat_copy.py` on the wd-old-model-runner branch.
# This is to ensure that we can still load older crepes, even if we change the
# dimensions of model layers, as we did with the Embedding layer in
# https://github.com/matsengrp/netam/pull/92.
def test_old_crepe_outputs():
    example_seq = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSSGMHWVRQAPGKGLEWVAVIWYDGSNKYYADSVKGRFTISRDNSKNTVYLQMNSLRAEDTAVYYCAREGHSNYPYYYYYMDVWGKGTTVTVSS"
    ddsm_crepe = load_crepe("tests/old_models/ddsm_13k-v1jaffe+v1tang-joint")
    dnsm_crepe = load_crepe("tests/old_models/dnsm_13k-v1jaffe+v1tang-joint")

    ddsm_vals = torch.nan_to_num(
        zap_predictions_along_diagonal(
            torch.load("tests/old_models/ddsm_output", weights_only=True).unsqueeze(0),
            aa_idx_tensor_of_str_ambig(example_seq).unsqueeze(0),
            fill=float("nan"),
        ).squeeze(0),
        0.0,
    )
    dnsm_vals = torch.load("tests/old_models/dnsm_output", weights_only=True)

    ddsm_result = torch.nan_to_num(ddsm_crepe([(example_seq, "")])[0][0], 0.0)
    dnsm_result = dnsm_crepe([(example_seq, "")])[0][0]
    assert torch.allclose(ddsm_result, ddsm_vals)
    assert torch.allclose(dnsm_result, dnsm_vals)
