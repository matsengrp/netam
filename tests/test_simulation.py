"""Pytest tests for simulation-related functions."""

import pytest
import torch
import pandas as pd

from netam.codon_table import CODON_AA_INDICATOR_MATRIX
from netam.pretrained import load_multihit
from netam.framework import (
    add_shm_model_outputs_to_pcp_df,
    codon_probs_of_parent_seq,
    trimmed_shm_model_outputs_of_crepe,
    sample_sequence_from_codon_probs,
    load_crepe,
)
from netam import pretrained
from netam.common import force_spawn, clamp_probability, clamp_log_probability
from netam.dasm import (
    DASMBurrito,
    DASMDataset,
)
from netam.dnsm import (
    DNSMBurrito,
    DNSMDataset,
)
import netam.molevol as molevol
from netam.molevol import (
    neutral_codon_probs_of_seq,
    zero_stop_codon_probs,
    set_parent_codon_prob,
)
from netam.models import DEFAULT_MULTIHIT_MODEL

from netam.models import TransformerBinarySelectionModelWiggleAct
import netam.sequences as sequences
from netam.sequences import (
    MAX_KNOWN_TOKEN_COUNT,
    aa_mask_tensor_of,
    nt_idx_tensor_of_str,
    translate_sequence,
    translate_sequence_mask_codons,
    translate_sequences,
    iter_codons,
    hamming_distance,
)
from test_dnsm import dnsm_burrito
from netam.codon_table import STOP_CODON_INDICATOR
from netam.hit_class import parent_specific_hit_classes


@pytest.fixture(scope="module")
def dasm_pred_burrito(pcp_df):
    force_spawn()
    """Fixture that returns the DASM Burrito object."""
    pcp_df["in_train"] = False

    model = TransformerBinarySelectionModelWiggleAct(
        nhead=2,
        d_model_per_head=4,
        dim_feedforward=256,
        layer_count=2,
        output_dim=20,
        model_type="dasm",
    )

    train_dataset, val_dataset = DASMDataset.train_val_datasets_of_pcp_df(
        pcp_df,
        MAX_KNOWN_TOKEN_COUNT,
        multihit_model=load_multihit(model.multihit_model_name),
    )
    train_dataset = val_dataset

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


@pytest.fixture(scope="module")
def dnsm_pred_burrito(pcp_df):
    force_spawn()
    """Fixture that returns the DNSM Burrito object."""
    pcp_df["in_train"] = False

    model = TransformerBinarySelectionModelWiggleAct(
        nhead=2,
        d_model_per_head=4,
        dim_feedforward=256,
        layer_count=2,
        model_type="dnsm",
    )

    train_dataset, val_dataset = DNSMDataset.train_val_datasets_of_pcp_df(
        pcp_df,
        MAX_KNOWN_TOKEN_COUNT,
        multihit_model=load_multihit(model.multihit_model_name),
    )
    train_dataset = val_dataset

    burrito = DNSMBurrito(
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
    dasm_pred_burrito.model.eval()
    print("updating neutral probs")
    dasm_pred_burrito.val_dataset.update_neutral_probs()
    burrito_preds = dasm_pred_burrito.val_dataset.log_neutral_codon_probss

    branch_lengths = dasm_pred_burrito.val_dataset.branch_lengths

    neutral_crepe = pretrained.load(dasm_pred_burrito.model.neutral_model_name)
    codon_probs = []
    hit_classes = []
    for (nt_parent, _), branch_length in zip(parent_seqs, branch_lengths):
        rates, csps = trimmed_shm_model_outputs_of_crepe(neutral_crepe, [nt_parent])
        mask = aa_mask_tensor_of(translate_sequence(nt_parent))
        hit_classes.append(
            parent_specific_hit_classes(
                nt_idx_tensor_of_str(nt_parent).reshape(-1, 3)
            ).view(-1, 64)
        )
        codon_probs.append(
            clamp_probability(
                set_parent_codon_prob(
                    neutral_codon_probs_of_seq(
                        nt_parent,
                        mask,
                        rates[0],
                        csps[0],
                        branch_length,
                        multihit_model=dasm_pred_burrito.val_dataset.multihit_model,
                    ),
                    torch.argmin(hit_classes[-1], dim=-1),
                )
            ).log()
        )
        # check that set_parent_codon_prob is idempotent on codon probs:
        adj_probs = set_parent_codon_prob(
            codon_probs[-1].clone().exp(), torch.argmin(hit_classes[-1], dim=-1)
        )
        if not torch.allclose(codon_probs[-1].exp(), adj_probs):
            adj_mask = ~torch.isclose(codon_probs[-1].exp(), adj_probs)
            print(adj_probs[adj_mask].detach().numpy())
            print(codon_probs[-1][adj_mask].exp().detach().numpy())
            print(
                (adj_probs - codon_probs[-1].exp())[adj_mask]
                .detach()
                .abs()
                .max()
                .numpy()
            )
            print(hit_classes[-1][adj_mask])
            assert False, "set_parent_codon_prob should be idempotent on codon probs"
    # Check that the predictions match
    for i, (pred, heavy_codon_prob, hit_classes) in enumerate(
        zip(burrito_preds, codon_probs, hit_classes)
    ):
        pred = clamp_log_probability(pred)
        pred = set_parent_codon_prob(
            pred[: len(heavy_codon_prob)].exp(), torch.argmin(hit_classes, dim=-1)
        ).log()
        wt_mask = hit_classes == 0
        # Because of log-instability close to 0, we compare wild type codon
        # probs on linear scale and all others in log scale.
        assert torch.allclose(
            pred[wt_mask].exp(), heavy_codon_prob[wt_mask].exp()
        ) and torch.allclose(pred[~wt_mask], heavy_codon_prob[~wt_mask])


def test_selection_probs(pcp_df, dasm_pred_burrito):
    """Test that the DASM burrito computes the same predictions as
    codon_probs_of_parent_seq."""
    # TO make the same test for dnsm, things are more complicated because the
    # burrito only produces aa-level probabilities.
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
    neutral_crepe = pretrained.load(dasm_pred_burrito.model.neutral_model_name)
    print("Computing from scratch")
    codon_probs = list(
        codon_probs_of_parent_seq(
            dasm_crepe,
            parent_seq,
            branch_length,
            neutral_crepe=neutral_crepe,
            multihit_model=dasm_pred_burrito.val_dataset.multihit_model,
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
            atol=1e-8,
        ), "Predictions should match"
        # Check that stop codons are zeroed out
        # netam.codon_table.STOP_CODON_INDICATOR is a length 64 tensor with 1s at stop codon indices
        if not torch.allclose(
            pred[:, STOP_CODON_INDICATOR == 1].exp(), torch.tensor(0.0)
        ):
            print(f"Stop codon probabilities are not zeroed out!")
            print(pred[:, STOP_CODON_INDICATOR == 1].exp())


def test_selection_probs_dnsm(pcp_df, dnsm_pred_burrito):
    """Test that the DNSM burrito computes the same predictions as
    codon_probs_of_parent_seq."""
    # TO make the same test for dnsm, things are more complicated because the
    # burrito only produces aa-level probabilities.
    parent_seqs = list(zip(pcp_df["parent_h"].tolist(), pcp_df["parent_l"].tolist()))

    print("recomputing branch lengths")
    dnsm_pred_burrito.standardize_and_optimize_branch_lengths()
    print("updating neutral probs")
    dnsm_pred_burrito.val_dataset.update_neutral_probs()
    # Get the predictions from the DNSM burrito
    dnsm_pred_burrito.batch_size = 500
    val_loader = dnsm_pred_burrito.build_val_loader()
    # There should be exactly one batch
    (batch,) = val_loader
    print("Getting predictions")
    burrito_preds = dnsm_pred_burrito.predictions_of_batch(batch)
    # ^ remember, this is one probability per-site of mutation.

    branch_lengths = dnsm_pred_burrito.val_dataset.branch_lengths

    # Get the predictions from codon_probs_of_parent_seq
    dnsm_crepe = dnsm_pred_burrito.to_crepe()
    neutral_crepe = pretrained.load(dnsm_pred_burrito.model.neutral_model_name)
    print("Computing from scratch")
    codon_probs_heavy = list(
        codon_probs_of_parent_seq(
            dnsm_crepe,
            parent_seq,
            branch_length,
            neutral_crepe=neutral_crepe,
            multihit_model=dnsm_pred_burrito.val_dataset.multihit_model,
        )[0]  # For DNSM, we need to use aa-level probabilities
        for parent_seq, branch_length in zip(parent_seqs, branch_lengths)
    )

    # Check that the predictions match
    for pred, heavy_codon_prob, parent_seq in zip(burrito_preds, codon_probs_heavy, parent_seqs):
        parent_codon_idxs = nt_idx_tensor_of_str(parent_seq[0]).reshape(-1, 3)
        hit_classes = parent_specific_hit_classes(parent_codon_idxs).view(-1, 64)
        no_mut_probs = heavy_codon_prob[hit_classes == 0].detach().clone()
        zeroed_wt_hcprob = heavy_codon_prob.detach().clone()
        zeroed_wt_hcprob[hit_classes == 0] = 0.0
        mut_sums = zeroed_wt_hcprob.sum(dim=-1)
        mut_probs = 1.0 - no_mut_probs
        if not torch.allclose(
            mut_probs, pred[: len(mut_probs)]
        ):
            print(f"The following should match:")
            print(mut_probs.detach().numpy())
            print(pred[: len(mut_probs)].detach().numpy())
            print("Difference:", (mut_probs - pred[: len(mut_probs)]).detach().numpy())

            print(f"The following should also match (not checked):")
            print(mut_sums.detach().numpy())
            print(pred[: len(mut_sums)].detach().numpy())
            print("Difference:", (mut_sums - pred[: len(mut_sums)]).detach().numpy())
            assert False

        if not torch.allclose(
            mut_sums, pred[:len(mut_sums)]
        ):
            print(f"The following should match:")
            print(mut_sums.detach().numpy())
            print(pred[: len(mut_sums)].detach().numpy())
            print("Difference:", (mut_sums - pred[: len(mut_sums)]).detach().numpy())
            assert False

        # heavy_codon_prob = heavy_codon_prob.log().type_as(pred)
        # print(pred[0].detach().numpy())
        # print(heavy_codon_prob[0].detach().numpy())
        # print(pred[0].logsumexp(0))
        # print(heavy_codon_prob[0].logsumexp(0))
        # print((pred[0] - heavy_codon_prob[0]).detach().numpy())
        # assert torch.allclose(
        #     # zero_stop_codon_probs(pred[: len(heavy_codon_prob)].exp()),
        #     pred[: len(heavy_codon_prob)].exp(),
        #     heavy_codon_prob.exp(),
        #     atol=1e-8,
        # ), "Predictions should match"
        # Check that stop codons are zeroed out
        # netam.codon_table.STOP_CODON_INDICATOR is a length 64 tensor with 1s at stop codon indices
        # if not torch.allclose(
        #     pred[:, STOP_CODON_INDICATOR == 1].exp(), torch.tensor(0.0)
        # ):
        #     print(f"Stop codon probabilities are not zeroed out!")
        #     print(pred[:, STOP_CODON_INDICATOR == 1].exp())


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
    neutral_crepe = pretrained.load(dasm_pred_burrito.model.neutral_model_name)

    # Process all sequences
    sequence_diffs = []
    per_sequence_stats = []

    for i, (parent_seq, branch_length) in enumerate(zip(parent_seqs, branch_lengths)):
        heavy_codon_probs, _ = codon_probs_of_parent_seq(
            dasm_crepe,
            parent_seq,
            branch_length,
            neutral_crepe=neutral_crepe,
            multihit_model=pretrained.load_multihit(dasm_crepe.model.multihit_model_name),
        )

        # Use only the heavy chain for sampling
        heavy_parent_seq = parent_seq[0]

        # Sample multiple sequences for this parent
        num_samples = 200
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


def test_refit_branch_lengths(pcp_df, dasm_pred_burrito):
    """Test that after simulating with a fixed branch length, branch length optimization recovers the original branch length, on average."""
    selection_crepe = dasm_pred_burrito.to_crepe()
    fixed_branch_length = 0.1
    replicates = 100
    multihit_model = load_multihit(selection_crepe.model.multihit_model_name)
    neutral_crepe = pretrained.load(selection_crepe.model.neutral_model_name)

    new_pcps = []
    for parent in pcp_df["parent_h"]:
        # Get the codon probabilities for the parent sequence
        heavy_codon_probs, _ = codon_probs_of_parent_seq(
            selection_crepe,
            (parent, ""),
            fixed_branch_length,
            neutral_crepe=neutral_crepe,
            multihit_model=multihit_model,
        )
        for _ in range(replicates):
            # Sample a sequence from the codon probabilities
            sampled_sequence = sample_sequence_from_codon_probs(heavy_codon_probs)

            if parent != sampled_sequence:
                new_pcps.append((parent, sampled_sequence))

    dataset_cls, burrito_cls = DASMDataset, DASMBurrito
    known_token_count = selection_crepe.model.hyperparameters["known_token_count"]
    neutral_crepe = pretrained.load(selection_crepe.model.neutral_model_name)
    multihit_model = pretrained.load_multihit(selection_crepe.model.multihit_model_name)
    new_pcp_df = pd.DataFrame(
        new_pcps,
        columns=["parent_h", "child_h"],
    )
    for col in pcp_df.columns:
        if col not in ["parent_h", "child_h"]:
            new_pcp_df[col] = list(pcp_df[col]) * replicates
    # Make val dataset from pcp_df:
    new_pcp_df = add_shm_model_outputs_to_pcp_df(new_pcp_df, neutral_crepe)
    new_pcp_df["in_train"] = False

    _, val_dataset = dataset_cls.train_val_datasets_of_pcp_df(
        new_pcp_df, known_token_count, multihit_model=multihit_model
    )

    burrito = burrito_cls(
        None,
        val_dataset,
        selection_crepe.model,
    )

    burrito.standardize_and_optimize_branch_lengths()
    assert torch.allclose(burrito.val_dataset.branch_lengths.mean().double(), torch.tensor(fixed_branch_length).double(), rtol=1e-2)


def test_selection_factors_with_crepe(pcp_df, dasm_pred_burrito):
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
    neutral_crepe = pretrained.load(dnsm_burrito.model.neutral_model_name)

    for i, (parent_seq, branch_length) in enumerate(zip(parent_seqs, branch_lengths)):
        heavy_codon_probs, _ = codon_probs_of_parent_seq(
            dnsm_crepe,
            parent_seq,
            branch_length,
            neutral_crepe=neutral_crepe,
            multihit_model=None,
        )

        sample_sequence_from_codon_probs(heavy_codon_probs)


def introduce_ns(sequence, site_p=0.05, seq_p=0.5):
    """Introduce N's into the sequence."""
    if torch.rand(1).item() > seq_p:
        return sequence
    # Convert the sequence to a list of characters
    seq_list = list(sequence)
    # Randomly select positions to introduce N's
    for i in range(len(seq_list)):
        if torch.rand(1).item() < site_p:  # 10% chance to introduce an N
            seq_list[i] = "N"
    # Convert back to a string
    return "".join(seq_list)


def test_ambig_sample_dnsm(pcp_df, dnsm_burrito):
    """Test that the DASM burrito can sample sequences with mutation counts similar to
    real data."""
    # Check that ambiguous sites are propagated to the child
    parent_seqs = list(zip(pcp_df["parent_h"].tolist(), pcp_df["parent_l"].tolist()))
    branch_lengths = dnsm_burrito.val_dataset.branch_lengths

    # Get the predictions from codon_probs_of_parent_seq
    dnsm_crepe = dnsm_burrito.to_crepe()
    neutral_crepe = pretrained.load(dnsm_burrito.model.neutral_model_name)

    for i, (parent_seq, branch_length) in enumerate(zip(parent_seqs, branch_lengths)):
        new_parent = tuple(introduce_ns(pseq) for pseq in parent_seq)
        heavy_codon_probs, _ = codon_probs_of_parent_seq(
            dnsm_crepe,
            new_parent,
            branch_length,
            neutral_crepe=neutral_crepe,
            multihit_model=None,
        )

        seq = sample_sequence_from_codon_probs(heavy_codon_probs)
        for i in range(2):
            for p, c in zip(iter_codons(new_parent[i]), iter_codons(seq[i])):
                if "N" in p:
                    assert c == "NNN", f"Codon {p} should be NNN, but got {c}"
                else:
                    assert "N" not in c, f"Codon {c} should not contain N, but got {p}"




# TODO these will need to be upgraded to dynamically created burritos to keep
# the test up to date. However, for speed I'll just read from disk:
def dasm_burrito(pcp_df):
    crepe = load_crepe("/home/wdumm/dnsm-netam-proj-runner1/dnsm-experiments-1/tests/models/dasm_13k-v1tangCC-joint")
    return DASMBurrito(
        None,
        DASMDataset.of_pcp_df(
            pcp_df,
            crepe.model.known_token_count,
            multihit_model=None,
        ),
        model=crepe.model,
    )


def dnsm_burrito(pcp_df):
    crepe = load_crepe("/home/wdumm/dnsm-netam-proj-runner1/dnsm-experiments-1/tests/models/dnsm_77k-v1jaffe50k-joint")
    return DNSMBurrito(
        None,
        DNSMDataset.of_pcp_df(
            pcp_df,
            crepe.model.known_token_count,
            multihit_model=None,
        ),
        model=crepe.model,
    )


def single_dnsm_burrito(pcp_df):
    crepe = load_crepe("/home/wdumm/dnsm-netam-proj-runner1/dnsm-experiments-1/dnsm-train/trained_models/single-tst-joint")
    return DNSMBurrito(
        None,
        DNSMDataset.of_pcp_df(
            pcp_df,
            crepe.model.known_token_count,
            multihit_model=None,
        ),
        model=crepe.model,
    )


@pytest.mark.parametrize("burrito_func", [single_dnsm_burrito, dasm_burrito, dnsm_burrito])
def test_selection_factors(pcp_df, burrito_func):
    burrito = burrito_func(pcp_df)
    # Make sure selection factors from the burrito match those from the crepe model:
    pcp_df = pcp_df.copy()
    pcp_df["parent_h"] = [introduce_ns(seq) for seq in pcp_df["parent_h"]]
    neutral_crepe = pretrained.load("ThriftyHumV0.2-59")
    pcp_df = add_shm_model_outputs_to_pcp_df(
        pcp_df.copy(),
        neutral_crepe,
    )
    for seq in pcp_df["parent_h"]:
        aa_parent = translate_sequence_mask_codons(seq)

        _token_nt_parent, _ = sequences.prepare_heavy_light_pair(
            seq,
            "",
            burrito.model.known_token_count,
        )
        _token_aa_parent = translate_sequence(_token_nt_parent)
        _token_aa_parent_idxs = sequences.aa_idx_tensor_of_str_ambig(_token_aa_parent)
        _token_aa_mask = sequences.codon_mask_tensor_of(_token_nt_parent)
        print("from burrito")
        sel_matrix = burrito.build_selection_matrix_from_parent_aa(_token_aa_parent_idxs, _token_aa_mask)[:len(aa_parent)]

        print("from crepe")
        from_crepe = burrito.to_crepe()([(aa_parent, "")])[0][0]
        if burrito.model.model_type == "dnsm":
            from_crepe = molevol.lift_to_per_aa_selection_factors(from_crepe, sequences.aa_idx_tensor_of_str_ambig(aa_parent))
        if not torch.allclose(from_crepe, sel_matrix):
            diff_mask = ~torch.isclose(from_crepe, sel_matrix, equal_nan=True)
            print("Differences in selection factors")
            print((from_crepe - sel_matrix)[diff_mask])
            print("from crepe:")
            print(from_crepe[diff_mask])
            print("From burrito:")
            print(sel_matrix[diff_mask])
            print("Parent sequence values at sites with differences")
            print("".join(char for char, m in zip(aa_parent, diff_mask.any(dim=-1).tolist()) if m))
            assert False


# TODO add dasm_burrito back
@pytest.mark.parametrize("burrito_func", [single_dnsm_burrito])
# @pytest.mark.parametrize("burrito_func", [dasm_burrito, dnsm_burrito])
def test_build_codon_mutsel(pcp_df, burrito_func):
    burrito = burrito_func(pcp_df)
    # There are two ways of computing codon probabilities. Let's make sure
    # they're the same:
    neutral_crepe = pretrained.load("ThriftyHumV0.2-59")
    pcp_df = pcp_df.copy()
    pcp_df["parent_h"] = [introduce_ns(seq) for seq in pcp_df["parent_h"]]
    pcp_df = add_shm_model_outputs_to_pcp_df(
        pcp_df.copy(),
        neutral_crepe,
    )
    multihit_model = pretrained.load_multihit(DEFAULT_MULTIHIT_MODEL)
    for multihit_model in [None, pretrained.load_multihit(DEFAULT_MULTIHIT_MODEL)]:
        branch_length = 0.06
        for seq, nt_rates, nt_csps in zip(pcp_df["parent_h"], pcp_df["nt_rates_h"], pcp_df["nt_csps_h"]):
            parent_idxs = sequences.nt_idx_tensor_of_str(seq.replace("N", "A"))
            aa_parent = translate_sequence(seq)
            codon_parent_idxs = sequences.codon_idx_tensor_of_str_ambig(seq)
            hit_classes = parent_specific_hit_classes(
                parent_idxs.reshape(-1, 3),
            )
            flat_hit_classes = molevol.flatten_codons(hit_classes)

            _token_nt_parent, _ = sequences.prepare_heavy_light_pair(
                seq,
                "",
                burrito.model.known_token_count,
            )
            _token_aa_parent = translate_sequence(_token_nt_parent)
            _token_aa_parent_idxs = sequences.aa_idx_tensor_of_str_ambig(_token_aa_parent)
            _token_aa_mask = sequences.codon_mask_tensor_of(_token_nt_parent)
            aa_mask = _token_aa_mask[:len(aa_parent)]
            # sel_matrix = torch.ones((aa_seq_len, 20))
            # This is in linear space.
            sel_matrix = burrito.build_selection_matrix_from_parent_aa(_token_aa_parent_idxs, _token_aa_mask)[:len(aa_parent)]
            # neutral_sel_matrix[torch.arange(aa_seq_len), aa_parent_idxs]

            # First way:
            nt_mut_probs = 1.0 - torch.exp(-branch_length * nt_rates)
            codon_mutsel, _ = molevol.build_codon_mutsel(
                parent_idxs.reshape(-1, 3),
                nt_mut_probs.reshape(-1, 3),  # Linear space
                nt_csps.reshape(-1, 3, 4),  # Linear space
                sel_matrix,  # Linear space
                multihit_model=multihit_model,
            )
            log_codon_mutsel = clamp_probability(codon_mutsel).log()
            flat_log_codon_mutsel = molevol.flatten_codons(log_codon_mutsel)
            flat_log_codon_mutsel[~aa_mask] = float("nan")

            # Second way:
            neutral_codon_probs = molevol.neutral_codon_probs_of_seq(
                seq,
                aa_mask,
                nt_rates,
                nt_csps,
                branch_length,
                multihit_model=multihit_model,
            )
            adjusted_codon_probs = molevol.adjust_codon_probs_by_aa_selection_factors(
                codon_parent_idxs.unsqueeze(0),
                neutral_codon_probs.unsqueeze(0).log(),
                sel_matrix.unsqueeze(0).log()
            ).squeeze(0)

            # Now let's compare to the simulation probs:
            sim_probs = clamp_probability(codon_probs_of_parent_seq(
                burrito.to_crepe(),
                (seq, ""),
                branch_length,
                neutral_crepe=neutral_crepe,
                multihit_model=multihit_model,
            )[0]).log()

            # adjusted_codon_probs = molevol.set_parent_codon_prob(adjusted_codon_probs.exp(), codon_parent_idxs,).log()
            # sim_probs = molevol.set_parent_codon_prob(sim_probs.exp(), codon_parent_idxs,).log()
            # flat_log_codon_mutsel = molevol.set_parent_codon_prob(flat_log_codon_mutsel.exp(), codon_parent_idxs,).log()
            # Compare mutsel path with adjust by selection factors path:
            if not torch.allclose(adjusted_codon_probs, flat_log_codon_mutsel, equal_nan=True, atol=1e-07):
                diff_mask = ~torch.isclose(adjusted_codon_probs, flat_log_codon_mutsel, equal_nan=True)
                print(flat_hit_classes[diff_mask])
                print((adjusted_codon_probs - flat_log_codon_mutsel)[diff_mask])
                print(adjusted_codon_probs[diff_mask])
                print(flat_log_codon_mutsel[diff_mask])
                assert False

            # adjusted_codon_probs = molevol.zero_stop_codon_probs(clamp_probability(adjusted_codon_probs.exp()).log())
            if not torch.allclose(adjusted_codon_probs, sim_probs, equal_nan=True, atol=1e-06):
                diff_mask = ~torch.isclose(adjusted_codon_probs, sim_probs, equal_nan=True)
                print(flat_hit_classes[diff_mask])
                print((adjusted_codon_probs - sim_probs)[diff_mask])
                print(adjusted_codon_probs[diff_mask])
                print(sim_probs[diff_mask])
                assert False
