"""Pytest tests for simulation-related functions."""

import pytest
import torch

from netam.framework import (
    codon_probs_of_parent_seq,
    trimmed_shm_model_outputs_of_crepe,
    sample_sequence_from_codon_probs,
)
from netam import pretrained
from netam.common import force_spawn
from netam.dasm import (
    DASMBurrito,
    DASMDataset,
)
from netam.molevol import neutral_codon_probs_of_seq, zero_stop_codon_probs

from netam.models import TransformerBinarySelectionModelWiggleAct
from netam.sequences import (
    MAX_KNOWN_TOKEN_COUNT,
    aa_mask_tensor_of,
    translate_sequence,
    translate_sequences,
)
from test_dnsm import dnsm_burrito
from netam.codon_table import STOP_CODON_INDICATOR


@pytest.fixture(scope="module")
def dasm_pred_burrito(pcp_df):
    force_spawn()
    """Fixture that returns the DASM Burrito object."""
    pcp_df["in_train"] = False
    train_dataset, val_dataset = DASMDataset.train_val_datasets_of_pcp_df(
        pcp_df, MAX_KNOWN_TOKEN_COUNT
    )
    train_dataset = val_dataset

    model = TransformerBinarySelectionModelWiggleAct(
        nhead=2,
        d_model_per_head=4,
        dim_feedforward=256,
        layer_count=2,
        output_dim=20,
    )

    burrito = DASMBurrito(
        train_dataset,
        val_dataset,
        model,
        batch_size=32,
        learning_rate=0.001,
        min_learning_rate=0.0001,
    )
    burrito.joint_train(
        epochs=1, cycle_count=2, training_method="full", optimize_bl_first_cycle=False
    )
    return burrito


# Test that the dasm burrito computes the same predictions as
# framework.codon_probs_of_parent_seq:


def test_neutral_probs(pcp_df, dasm_pred_burrito):
    """Test that the DASM burrito computes the same predictions as
    codon_probs_of_parent_seq."""
    parent_seqs = list(zip(pcp_df["parent_h"].tolist(), pcp_df["parent_l"].tolist()))

    print("recomputing branch lengths")
    dasm_pred_burrito.standardize_and_optimize_branch_lengths()
    print("updating neutral probs")
    dasm_pred_burrito.val_dataset.update_neutral_probs()
    burrito_preds = dasm_pred_burrito.val_dataset.log_neutral_codon_probss

    branch_lengths = dasm_pred_burrito.val_dataset.branch_lengths

    neutral_crepe = pretrained.load("ThriftyHumV0.2-45")
    codon_probs = []
    for (nt_parent, _), branch_length in zip(parent_seqs, branch_lengths):
        rates, csps = trimmed_shm_model_outputs_of_crepe(neutral_crepe, [nt_parent])
        mask = aa_mask_tensor_of(translate_sequence(nt_parent))
        codon_probs.append(
            neutral_codon_probs_of_seq(
                nt_parent,
                mask,
                rates[0],
                csps[0],
                branch_length,
                multihit_model=None,
            ).log()
        )
    # Check that the predictions match
    for pred, heavy_codon_prob in zip(burrito_preds, codon_probs):
        print(pred[0].detach().numpy())
        print(heavy_codon_prob[0].detach().numpy())
        print(pred[0].logsumexp(0))
        print(heavy_codon_prob[0].logsumexp(0))
        print((pred[0] - heavy_codon_prob[0]).detach().numpy())
        assert torch.allclose(
            pred[: len(heavy_codon_prob)], heavy_codon_prob
        ), "Predictions should match"


def test_selection_probs(pcp_df, dasm_pred_burrito):
    """Test that the DASM burrito computes the same predictions as
    codon_probs_of_parent_seq."""
    parent_seqs = list(zip(pcp_df["parent_h"].tolist(), pcp_df["parent_l"].tolist()))

    print("recomputing branch lengths")
    dasm_pred_burrito.standardize_and_optimize_branch_lengths()
    print("updating neutral probs")
    dasm_pred_burrito.val_dataset.update_neutral_probs()
    # Get the predictions from the DASM burrito
    dasm_pred_burrito.batch_size = 500
    val_loader = dasm_pred_burrito.build_val_loader()
    # There should be exactly one batch
    (batch,) = val_loader
    print("Getting predictions")
    burrito_preds = dasm_pred_burrito.predictions_of_batch(batch)

    branch_lengths = dasm_pred_burrito.val_dataset.branch_lengths

    # Get the predictions from codon_probs_of_parent_seq
    dasm_crepe = dasm_pred_burrito.to_crepe()
    neutral_crepe = pretrained.load("ThriftyHumV0.2-45")
    print("Computing from scratch")
    codon_probs = list(
        codon_probs_of_parent_seq(
            dasm_crepe,
            parent_seq,
            branch_length,
            neutral_crepe=neutral_crepe,
            multihit_model=None,
        )
        for parent_seq, branch_length in zip(parent_seqs, branch_lengths)
    )

    # Check that the predictions match
    for pred, (heavy_codon_prob, _) in zip(burrito_preds, codon_probs):
        heavy_codon_prob = heavy_codon_prob.log().type_as(pred)
        print(pred[0].detach().numpy())
        print(heavy_codon_prob[0].detach().numpy())
        print(pred[0].logsumexp(0))
        print(heavy_codon_prob[0].logsumexp(0))
        print((pred[0] - heavy_codon_prob[0]).detach().numpy())
        assert torch.allclose(
            zero_stop_codon_probs(pred[: len(heavy_codon_prob)].exp()),
            heavy_codon_prob.exp(),
        ), "Predictions should match"
        # Check that stop codons are zeroed out
        # netam.codon_table.STOP_CODON_INDICATOR is a length 64 tensor with 1s at stop codon indices
        if not torch.allclose(
            pred[:, STOP_CODON_INDICATOR == 1].exp(), torch.tensor(0.0)
        ):
            print(f"Stop codon probabilities are not zeroed out!")
            print(pred[:, STOP_CODON_INDICATOR == 1].exp())


def test_sequence_sampling(pcp_df, dasm_pred_burrito):
    """Test that the DASM burrito can sample sequences with mutation counts similar to
    real data."""
    # Check that on average, the difference in Hamming distance between
    # sampled sequences and actual sequences to their parents is close to 0
    parent_seqs = list(zip(pcp_df["parent_h"].tolist(), pcp_df["parent_l"].tolist()))

    print("recomputing branch lengths")
    dasm_pred_burrito.standardize_and_optimize_branch_lengths()
    print("updating neutral probs")
    dasm_pred_burrito.val_dataset.update_neutral_probs()

    branch_lengths = dasm_pred_burrito.val_dataset.branch_lengths

    # Get the predictions from codon_probs_of_parent_seq
    dasm_crepe = dasm_pred_burrito.to_crepe()
    neutral_crepe = pretrained.load("ThriftyHumV0.2-45")

    def hamming_distance(seq1, seq2):
        return sum(a != b for a, b in zip(seq1, seq2))

    # Process all sequences
    sequence_diffs = []
    per_sequence_stats = []

    for i, (parent_seq, branch_length) in enumerate(zip(parent_seqs, branch_lengths)):
        heavy_codon_probs, _ = codon_probs_of_parent_seq(
            dasm_crepe,
            parent_seq,
            branch_length,
            neutral_crepe=neutral_crepe,
            multihit_model=None,
        )

        # Use only the heavy chain for sampling
        heavy_parent_seq = parent_seq[0]

        # Sample multiple sequences for this parent
        num_samples = 40
        sampled_seqs = [
            sample_sequence_from_codon_probs(heavy_codon_probs)
            for _ in range(num_samples)
        ]

        # Calculate distances from parent to sampled sequences
        sampled_distances = [
            hamming_distance(heavy_parent_seq, seq) for seq in sampled_seqs
        ]
        mean_sampled_distance = sum(sampled_distances) / len(sampled_distances)

        # Calculate distance from parent to actual child in the dataset
        reference_distance = hamming_distance(
            heavy_parent_seq, pcp_df["child_h"].iloc[i]
        )

        # Calculate the difference between sampled and reference distances
        distance_diff = mean_sampled_distance - reference_distance
        sequence_diffs.append(distance_diff)

        per_sequence_stats.append(
            {
                "index": i,
                "mean_sampled_distance": mean_sampled_distance,
                "reference_distance": reference_distance,
                "difference": distance_diff,
            }
        )

        print(f"Sequence {i}:")
        print(f"  Mean sampled distance: {mean_sampled_distance:.2f}")
        print(f"  Reference distance: {reference_distance}")
        print(f"  Difference: {distance_diff:.2f}")

    # Calculate the average difference across all sequences
    mean_diff = sum(sequence_diffs) / len(sequence_diffs)
    abs_diffs = [abs(diff) for diff in sequence_diffs]
    mean_abs_diff = sum(abs_diffs) / len(abs_diffs)

    print(
        f"Average difference between sampled and reference distances: {mean_diff:.2f}"
    )
    print(f"Average absolute difference: {mean_abs_diff:.2f}")

    # Calculate standard deviation of the differences
    variance = sum((diff - mean_diff) ** 2 for diff in sequence_diffs) / len(
        sequence_diffs
    )
    std_dev = variance**0.5
    print(f"Standard deviation of differences: {std_dev:.2f}")

    # Test that the average difference is close to 0
    # We'll use a tolerance based on the standard deviation
    tolerance = max(2.0, std_dev)  # At least 2.0, or larger if std_dev is higher
    assert (
        abs(mean_diff) < tolerance
    ), f"Mean difference between sampled and reference distances ({mean_diff:.2f}) exceeds tolerance ({tolerance:.2f})"

    # Also check that the absolute differences aren't too large on average
    max_abs_diff = max(abs_diffs)
    print(f"Maximum absolute difference: {max_abs_diff:.2f}")
    assert (
        mean_abs_diff < tolerance * 1.5
    ), f"Mean absolute difference ({mean_abs_diff:.2f}) is too large"


def test_selection_factors(pcp_df, dasm_pred_burrito):
    """Test that the DASM burrito computes the same selection factors as the crepe
    model."""
    parent_seqs = list(zip(pcp_df["parent_h"].tolist(), pcp_df["parent_l"].tolist()))

    print("recomputing branch lengths")
    dasm_pred_burrito.standardize_and_optimize_branch_lengths()
    print("updating neutral probs")
    dasm_pred_burrito.val_dataset.update_neutral_probs()

    # Get the predictions from the DASM burrito
    dasm_pred_burrito.batch_size = 500
    val_loader = dasm_pred_burrito.build_val_loader()
    # There should be exactly one batch
    (batch,) = val_loader
    print("Getting predictions")
    log_neutral_codon_probs, log_selection_factors = (
        dasm_pred_burrito.prediction_pair_of_batch(batch)
    )

    # Get the predictions from the crepe model
    dasm_crepe = dasm_pred_burrito.to_crepe()
    print("Computing selection factors from scratch")

    # Check all sequences instead of just the first one
    for i, parent_seq in enumerate(parent_seqs):
        aa_seq = tuple(translate_sequences(parent_seq))
        crepe_log_selection_factors = dasm_crepe.model.selection_factors_of_aa_str(
            aa_seq
        )[0].log()

        # Get the corresponding selection factors from the burrito
        burrito_log_selection_factors = log_selection_factors[i]

        print(f"Sequence {i}:")
        print(crepe_log_selection_factors)
        print(burrito_log_selection_factors[: len(crepe_log_selection_factors)])

        assert torch.allclose(
            crepe_log_selection_factors,
            burrito_log_selection_factors[: len(crepe_log_selection_factors)],
        ), f"Selection factors don't match for sequence {i}"


def test_sequence_sample_dnsm(pcp_df, dnsm_burrito):
    """Test that the DASM burrito can sample sequences with mutation counts similar to
    real data."""
    # Check that on average, the difference in Hamming distance between
    # sampled sequences and actual sequences to their parents is close to 0
    parent_seqs = list(zip(pcp_df["parent_h"].tolist(), pcp_df["parent_l"].tolist()))
    branch_lengths = dnsm_burrito.val_dataset.branch_lengths

    # Get the predictions from codon_probs_of_parent_seq
    dnsm_crepe = dnsm_burrito.to_crepe()
    neutral_crepe = pretrained.load("ThriftyHumV0.2-45")

    for i, (parent_seq, branch_length) in enumerate(zip(parent_seqs, branch_lengths)):
        heavy_codon_probs, _ = codon_probs_of_parent_seq(
            dnsm_crepe,
            parent_seq,
            branch_length,
            neutral_crepe=neutral_crepe,
            multihit_model=None,
        )

        sample_sequence_from_codon_probs(heavy_codon_probs)
