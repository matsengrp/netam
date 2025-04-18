import os
from collections import Counter

import netam.multihit as multihit
import netam.molevol as molevol
import netam.framework as framework
import netam.hit_class as hit_class
from netam.molevol import (
    codon_probs_of_parent_scaled_nt_rates_and_csps,
    reshape_for_codons,
    neutral_codon_probs,
)
from netam.codon_table import (
    build_stop_codon_indicator_tensor,
)
from scipy import optimize
from netam import pretrained
from netam.common import clamp_probability, clamp_log_probability, BIG
from netam.sequences import (
    nt_idx_tensor_of_str,
    MAX_KNOWN_TOKEN_COUNT,
    codon_idx_tensor_of_str_ambig,
    nt_idx_tensor_of_str,
    iter_codons,
    STOP_CODONS,
    aa_idx_tensor_of_str_ambig,
    apply_aa_mask_to_nt_sequence,
    translate_sequence,
)
from netam.models import SingleValueBinarySelectionModel, HitClassModel
from netam.dasm import DASMDataset, DASMBurrito
from netam.dnsm import DNSMDataset, DNSMBurrito
from netam.hit_class import parent_specific_hit_classes
import pytest
import pandas as pd
import torch

burrito_params = {
    "batch_size": 1024,
    "learning_rate": 0.1,
    "min_learning_rate": 1e-4,
}


# These happen to be the same as some examples in test_models.py but that's fine.
# If it was important that they were shared, we should put them in a conftest.py.
ex_scaled_rates = torch.tensor([0.01, 0.001, 0.005])
ex_csps = torch.tensor(
    [[0.0, 0.3, 0.5, 0.2], [0.4, 0.0, 0.1, 0.5], [0.2, 0.3, 0.0, 0.5]]
)
# This is an example, and the correct output for test_codon_probs_of_parent_scaled_nt_rates_and_csps
ex_codon_probs = torch.tensor(
    [
        [
            [
                [3.9484e-07, 5.9226e-07, 3.9385e-04, 9.8710e-07],
                [9.8660e-04, 1.4799e-03, 9.8413e-01, 2.4665e-03],
                [9.8710e-08, 1.4806e-07, 9.8463e-05, 2.4677e-07],
                [4.9355e-07, 7.4032e-07, 4.9231e-04, 1.2339e-06],
            ],
            [
                [1.1905e-09, 1.7857e-09, 1.1875e-06, 2.9762e-09],
                [2.9746e-06, 4.4619e-06, 2.9672e-03, 7.4366e-06],
                [2.9762e-10, 4.4642e-10, 2.9687e-07, 7.4404e-10],
                [1.4881e-09, 2.2321e-09, 1.4844e-06, 3.7202e-09],
            ],
            [
                [1.9841e-09, 2.9762e-09, 1.9791e-06, 4.9602e-09],
                [4.9577e-06, 7.4366e-06, 4.9453e-03, 1.2394e-05],
                [4.9602e-10, 7.4404e-10, 4.9478e-07, 1.2401e-09],
                [2.4801e-09, 3.7202e-09, 2.4739e-06, 6.2003e-09],
            ],
            [
                [7.9364e-10, 1.1905e-09, 7.9165e-07, 1.9841e-09],
                [1.9831e-06, 2.9746e-06, 1.9781e-03, 4.9577e-06],
                [1.9841e-10, 2.9762e-10, 1.9791e-07, 4.9602e-10],
                [9.9205e-10, 1.4881e-09, 9.8957e-07, 2.4801e-09],
            ],
        ]
    ]
)

ex_parent_codon_idxs = nt_idx_tensor_of_str("ACG")


@pytest.fixture
def mini_multihit_train_val_datasets():
    df = pd.read_csv("data/wyatt-10x-1p5m_pcp_2023-11-30_NI.first100.csv.gz")
    crepe = pretrained.load("ThriftyHumV0.2-45")
    df = multihit.prepare_pcp_df(df, crepe, 500)
    return multihit.train_test_datasets_of_pcp_df(df)


@pytest.fixture
def hitclass_burrito(mini_multihit_train_val_datasets):
    train_data, val_data = mini_multihit_train_val_datasets
    return multihit.MultihitBurrito(
        train_data, val_data, multihit.HitClassModel(), **burrito_params
    )


def test_train(hitclass_burrito):
    before_values = hitclass_burrito.model.values.clone()
    hitclass_burrito.joint_train(epochs=2)
    assert not torch.allclose(hitclass_burrito.model.values, before_values)


def test_serialize(hitclass_burrito):
    os.makedirs("_ignore", exist_ok=True)
    hitclass_burrito.save_crepe("_ignore/test_multihit_crepe")
    new_crepe = framework.load_crepe("_ignore/test_multihit_crepe")
    assert torch.allclose(new_crepe.model.values, hitclass_burrito.model.values)


def test_codon_probs_of_parent_scaled_nt_rates_and_csps():
    computed_tensor = codon_probs_of_parent_scaled_nt_rates_and_csps(
        ex_parent_codon_idxs, ex_scaled_rates, ex_csps
    )
    correct_tensor = ex_codon_probs
    assert torch.allclose(correct_tensor, computed_tensor)
    assert torch.allclose(
        computed_tensor.sum(dim=(1, 2, 3)), torch.ones(computed_tensor.shape[0])
    )


def test_multihit_correction():
    hit_class_factors = torch.tensor([-0.1, 1, 2.3])
    # We'll verify that aggregating by hit class then adjusting is the same as adjusting then aggregating by hit class.
    codon_idxs = reshape_for_codons(ex_parent_codon_idxs)
    adjusted_codon_probs = hit_class.apply_multihit_correction(
        codon_idxs, ex_codon_probs, hit_class_factors
    )
    aggregate_last = hit_class.hit_class_probs_tensor(codon_idxs, adjusted_codon_probs)

    uncorrected_hc_log_probs = hit_class.hit_class_probs_tensor(
        codon_idxs, ex_codon_probs
    ).log()

    corrections = torch.cat([torch.tensor([0.0]), hit_class_factors])
    # we'll use the corrections to adjust the uncorrected hit class probs
    corrections = corrections[
        torch.arange(4).unsqueeze(0).tile((uncorrected_hc_log_probs.shape[0], 1))
    ]
    uncorrected_hc_log_probs += corrections
    aggregate_first = torch.softmax(uncorrected_hc_log_probs, dim=1)
    assert torch.allclose(aggregate_first, aggregate_last)


def test_hit_class_tensor():
    # verify that the opaque way of computing the hit class tensor is the same
    # as the transparent way.
    def compute_hit_class(codon1, codon2):
        return sum(c1 != c2 for c1, c2 in zip(codon1, codon2))

    true_hit_class_tensor = torch.zeros(4, 4, 4, 4, 4, 4, dtype=torch.int)

    # Populate the tensor
    for i1 in range(4):
        for j1 in range(4):
            for k1 in range(4):
                codon1 = (i1, j1, k1)
                for i2 in range(4):
                    for j2 in range(4):
                        for k2 in range(4):
                            codon2 = (i2, j2, k2)
                            true_hit_class_tensor[i1, j1, k1, i2, j2, k2] = (
                                compute_hit_class(codon1, codon2)
                            )
    assert torch.allclose(hit_class.hit_class_tensor, true_hit_class_tensor)


def make_dasm_burrito(multihit_model, pcp_df):
    dataset = DASMDataset.of_pcp_df(
        pcp_df, MAX_KNOWN_TOKEN_COUNT, multihit_model=multihit_model
    )
    model = SingleValueBinarySelectionModel(
        output_dim=20,
        known_token_count=MAX_KNOWN_TOKEN_COUNT,
    )
    model.single_value = torch.nn.Parameter(torch.tensor(0.0))
    burrito = DASMBurrito(
        dataset,
        dataset,
        model,
        batch_size=200,
        learning_rate=0.001,
        min_learning_rate=0.0001,
    )
    return burrito


def make_dnsm_burrito(multihit_model, pcp_df):
    dataset = DNSMDataset.of_pcp_df(
        pcp_df, MAX_KNOWN_TOKEN_COUNT, multihit_model=multihit_model
    )
    model = SingleValueBinarySelectionModel()
    model.single_value = torch.nn.Parameter(torch.tensor(0.0))
    burrito = DNSMBurrito(
        dataset,
        dataset,
        model,
        batch_size=200,
        learning_rate=0.001,
        min_learning_rate=0.0001,
    )
    return burrito


def make_test_pcp_df(len_factor=1, extend_same=0):
    # Here we have a hit class 0 codon and a hit class 1 codon in each pair,
    # and one pair each with a 0, 1, 2, and 3 hit class codon.

    # This just repeats the sequence the specified number of times. Seems like
    # that shouldn't change the branch length, but it does.
    _parent_seq = "ATGTACTTA"
    _child_seqs = ["ATGTACTCA", "ATGTATTCA", "ATGTTGTCA", "ATGGTGTCA"]
    parent_seq = _parent_seq * len_factor + _parent_seq * extend_same
    child_seqs = [
        (child * len_factor) + (_parent_seq * extend_same) for child in _child_seqs
    ]
    df = pd.DataFrame(
        {"parent": [parent_seq] * 4, "child": child_seqs, "v_gene": ["IGHV1-39*01"] * 4}
    )
    for child in child_seqs:
        print(sum(c1 != c2 for c1, c2 in zip(parent_seq, child)))
    # df = pd.DataFrame({"parent": ["ATGTAC"] * 3, "child": ["ATGTAT", "ATGTTG", "ATGGTG"], "v_gene": ["IGHV1-39*01"] * 3})
    pcp_df = framework.standardize_heavy_light_columns(df)
    pcp_df = framework.add_shm_model_outputs_to_pcp_df(
        pcp_df, pretrained.load("ThriftyHumV0.2-45")
    )
    # Neutralize neutral rates
    pcp_df["nt_rates_h"] = [torch.tensor([1.0] * len(parent_seq))] * len(pcp_df)
    nt_csps = list(pcp_df["nt_csps_h"])
    for i in range(len(nt_csps)):
        val = nt_csps[i]
        val[val > 0.0] = 1.0 / 3.0
    pcp_df["nt_csps_h"] = nt_csps
    return pcp_df


def simple_codon_probs_of_seqs(nt_parent, branch_length=1.0):
    nt_idx_parent = nt_idx_tensor_of_str(nt_parent)
    mut_probs = 1.0 - torch.exp(-branch_length * torch.full((len(nt_parent),), 1.0))
    csps = torch.full((len(nt_parent), 4), 1.0 / 3.0)
    csps[torch.arange(len(nt_parent)), nt_idx_parent] = 0.0

    neutral_probs = neutral_codon_probs(
        nt_idx_parent.reshape(-1, 3), mut_probs.reshape(-1, 3), csps.reshape(-1, 3, 4)
    )
    return neutral_probs


_stop_zapper_lin = build_stop_codon_indicator_tensor()


def simple_stop_zapped_codon_probs_of_seqs(nt_parent, branch_length=1.0):
    nt_idx_parent = nt_idx_tensor_of_str(nt_parent)
    neutral_probs = simple_codon_probs_of_seqs(nt_parent, branch_length)
    # Sum the stop codon probabilities and add them to the parent codon
    # probabilities.
    stop_probs = neutral_probs[:, _stop_zapper_lin.bool()].sum(dim=1)
    parent_hit_classes = parent_specific_hit_classes(nt_idx_parent.reshape(-1, 3))
    parent_codons = parent_hit_classes == 0
    neutral_probs[parent_codons.view(-1, 64)] += stop_probs
    neutral_probs[:, _stop_zapper_lin.bool()] = 0.0
    # neutral_probs = clamp_probability(neutral_probs)
    # neutral_probs = neutral_probs.log()
    # neutral_probs += (_stop_zapper_lin * -BIG)

    return neutral_probs


def _codon_prob_of_hit_class(hit_class, branch_length=torch.tensor(1.0), rate=1.0):
    # Assuming csp of 1/3 for muts
    prob_of_site_mutation = 1.0 - torch.exp(torch.tensor(-branch_length * rate))
    prob_of_non_mutation = 1.0 - prob_of_site_mutation
    return (prob_of_non_mutation ** (3 - hit_class)) * (
        (0.33333333 * prob_of_site_mutation) ** hit_class
    )


def _codon_prob_of_hit_class_stop_zapped(
    hit_class, parent_codon, branch_length=torch.tensor(1.0), rate=1.0
):
    assert parent_codon not in STOP_CODONS
    prob = _codon_prob_of_hit_class(hit_class, branch_length=branch_length, rate=rate)
    if hit_class == 0:
        # Add probabilities of all the stop codons to prob.
        for sc in STOP_CODONS:
            prob += _codon_prob_of_hit_class(
                _hamming_dist(sc, parent_codon), branch_length=branch_length, rate=rate
            )
    return prob


def test_codon_probs():
    nt_parent = "ATGTACTTA"
    nt_child = "ATGGTGTCA"
    nt_idx_parent = nt_idx_tensor_of_str(nt_parent)

    neutral_probs = simple_codon_probs_of_seqs(nt_parent)
    print(neutral_probs)
    parent_hit_classes = parent_specific_hit_classes(nt_idx_parent.reshape(-1, 3))
    # print(parent_hit_classes.view(-1, 64))

    test_codon_probs = torch.tensor([_codon_prob_of_hit_class(i) for i in range(4)])
    assert torch.allclose(
        neutral_probs, test_codon_probs[parent_hit_classes.view(-1, 64)]
    )

    # now test molevol.build_codon_mutsel:


def test_codon_probs_of_burrito():
    pcp_df = make_test_pcp_df()
    # branch_lengths = torch.tensor([0.1379, 0.2704, 0.4372, 0.6630])
    branch_length = 0.6
    branch_lengths = torch.tensor([branch_length] * len(pcp_df))
    no_mh_burrito = make_dasm_burrito(None, pcp_df)
    optimization_kwargs = {"learning_rate": 0.01, "optimization_tol": 1e-3}
    ds = no_mh_burrito.val_dataset
    ds.branch_lengths = branch_lengths
    ds.update_neutral_probs()
    burrito_neutral_probs = ds.log_neutral_codon_probss

    # Now test predictions_of_batch
    val_loader = no_mh_burrito.build_val_loader()
    (batch,) = val_loader

    predictions = no_mh_burrito.predictions_of_batch(batch)

    for row, burrito_probs, prediction_probs in zip(
        pcp_df.itertuples(), burrito_neutral_probs, predictions
    ):
        nt_parent = row.parent_h
        nt_idx_parent = nt_idx_tensor_of_str(nt_parent)
        parent_hit_classes = parent_specific_hit_classes(nt_idx_parent.reshape(-1, 3))
        test_probs = simple_codon_probs_of_seqs(nt_parent, branch_length=branch_length)
        if not torch.allclose(test_probs, burrito_probs[: len(test_probs)].exp()):
            print(test_probs.log().exp() - burrito_probs[: len(test_probs)].exp())
            print(parent_hit_classes.view(-1, 64))
            assert False

        test_probs = simple_stop_zapped_codon_probs_of_seqs(
            nt_parent, branch_length=branch_length
        )
        if not torch.allclose(
            test_probs.log().exp(),
            prediction_probs[: len(test_probs)].exp(),
            atol=1.2e-07,
        ):
            print(test_probs.log().exp() - prediction_probs[: len(test_probs)].exp())
            print(parent_hit_classes.view(-1, 64))
            assert False
        # Test molevol.build_codon_mutsel:
        codon_mutsel_probs, _ = molevol.build_codon_mutsel(
            nt_idx_parent.reshape(-1, 3),
            (1.0 - torch.exp(-branch_length * row.nt_rates_h.reshape(-1, 3))),
            row.nt_csps_h.reshape(-1, 3, 4),
            torch.full((len(nt_parent) // 3, 20), 1.0),
            multihit_model=None,
        )
        if not torch.allclose(
            test_probs, codon_mutsel_probs[: len(test_probs)].view(-1, 64)
        ):
            print(test_probs.log().exp() - codon_mutsel_probs[: len(test_probs)].exp())
            print(parent_hit_classes.view(-1, 64))
            assert False


def _hamming_dist(seq1, seq2):
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


def _hit_classes_of_seqs(nt_parent, nt_child):
    for c1, c2 in zip(iter_codons(nt_parent), iter_codons(nt_child)):
        yield _hamming_dist(c1, c2)


def _prob_of_branch_simplest(nt_parent, nt_child):
    observed_hit_classes = list(_hit_classes_of_seqs(nt_parent, nt_child))
    print(observed_hit_classes)

    def branch_prob(branch_length):
        val = sum(
            _codon_prob_of_hit_class(hc, branch_length=branch_length).log()
            for hc in observed_hit_classes
        )
        return val

    return branch_prob


def _prob_of_branch_simplest(nt_parent, nt_child):
    observed_hit_classes = list(_hit_classes_of_seqs(nt_parent, nt_child))
    print(observed_hit_classes)

    def branch_prob(branch_length):
        val = sum(
            _codon_prob_of_hit_class(hc, branch_length=branch_length).log()
            for hc in observed_hit_classes
        )
        return val

    return branch_prob


def _prob_of_branch_simple_zapped(nt_parent, nt_child):
    observed_hit_classes = list(_hit_classes_of_seqs(nt_parent, nt_child))
    print(observed_hit_classes)

    def branch_prob(branch_length):
        val = sum(
            _codon_prob_of_hit_class_stop_zapped(
                hc, cod, branch_length=branch_length
            ).log()
            for hc, cod in zip(observed_hit_classes, iter_codons(nt_parent))
        )
        return val

    return branch_prob


def _prob_of_branch(nt_parent, nt_child):
    observed_hit_classes = list(_hit_classes_of_seqs(nt_parent, nt_child))
    child_codons = nt_idx_tensor_of_str(nt_child).view(-1, 3)

    def branch_prob(branch_length):
        codon_probs = simple_stop_zapped_codon_probs_of_seqs(
            nt_parent, branch_length=branch_length
        ).view(-1, 4, 4, 4)
        observed_probs = codon_probs[
            torch.arange(len(child_codons)),
            child_codons[:, 0],
            child_codons[:, 1],
            child_codons[:, 2],
        ]
        return observed_probs.log().sum()

    return branch_prob


def _prob_of_branch_mutsel_log_pcp_prob(nt_parent, nt_child):
    aa_parent_str = translate_sequence(nt_parent)
    aa_parents_indices = aa_idx_tensor_of_str_ambig(aa_parent_str)
    aa_mask = torch.full_like(aa_parents_indices, True).bool()
    sel_matrix = torch.full((len(aa_parents_indices), 20), 1.0)
    nt_rates = torch.full((len(nt_parent),), 1.0)
    nt_csps = torch.full((len(nt_parent), 4), 1.0 / 3.0)
    nt_csps[torch.arange(len(nt_parent)), nt_idx_tensor_of_str(nt_parent)] = 0.0
    # Masks may be padded at end to account for sequences of different
    # lengths. The first part of the mask up to parent length should be
    # all the valid bits for the sequence.
    trimmed_aa_mask = aa_mask[: len(nt_parent)]
    log_pcp_probability = molevol.mutsel_log_pcp_probability_of(
        sel_matrix[aa_mask],
        apply_aa_mask_to_nt_sequence(nt_parent, trimmed_aa_mask),
        apply_aa_mask_to_nt_sequence(nt_child, trimmed_aa_mask),
        nt_rates[aa_mask.repeat_interleave(3)],
        nt_csps[aa_mask.repeat_interleave(3)],
        None,
    )
    return lambda x: log_pcp_probability(torch.log(torch.tensor(x)))


def _fit_branch_length(nt_parent, nt_child, prob_func_func):
    # branch_prob = _prob_of_branch(nt_parent, nt_child)
    branch_prob = prob_func_func(nt_parent, nt_child)

    result = optimize.minimize_scalar(
        lambda x: -branch_prob(x).item(),
        bounds=(0.0, 1.0),
        method="bounded",
    )

    return result.x


def test_manual_branch_lengths():
    pcp_df = make_test_pcp_df(1)
    validated_branch_lengths = [
        _fit_branch_length(parent, child, _prob_of_branch_simple_zapped)
        for parent, child in zip(pcp_df["parent_h"], pcp_df["child_h"])
    ]
    branch_lengths_from_used_func = [
        _fit_branch_length(parent, child, _prob_of_branch_mutsel_log_pcp_prob)
        for parent, child in zip(pcp_df["parent_h"], pcp_df["child_h"])
    ]
    if not torch.allclose(
        torch.tensor(validated_branch_lengths),
        torch.tensor(branch_lengths_from_used_func),
    ):
        print(validated_branch_lengths)
        print(branch_lengths_from_used_func)
        assert False


def test_pcp_prob_dxsm():
    len_factor = 1
    pcp_df = make_test_pcp_df(len_factor=len_factor)

    validated_no_mh_lengths = torch.tensor(
        [
            _fit_branch_length(parent, child, _prob_of_branch_simple_zapped)
            for parent, child in zip(pcp_df["parent_h"], pcp_df["child_h"])
        ]
    )

    no_mh_burrito = make_dasm_burrito(None, pcp_df)

    for parent, child, nt_rates, nt_csps in zip(
        pcp_df["parent_h"], pcp_df["child_h"], pcp_df["nt_rates_h"], pcp_df["nt_csps_h"]
    ):
        sample_branch_length = torch.tensor(0.5)
        val_prob_func = _prob_of_branch_simple_zapped(
            parent,
            child,
        )
        val_prob = val_prob_func(sample_branch_length)
        check_prob_func = _prob_of_branch_mutsel_log_pcp_prob(parent, child)
        check_prob = check_prob_func(sample_branch_length)
        print()
        if not torch.allclose(torch.tensor(check_prob), torch.tensor(val_prob)):
            print("len_adjusted check_prob", check_prob / len_factor)
            print("len_adjusted val_prob", val_prob / len_factor)
            print("parent", parent)
            print("child", child)
            assert False


def test_multihit_branch_lengths_dasm():
    pcp_df = make_test_pcp_df(len_factor=1, extend_same=39)

    validated_no_mh_lengths = torch.tensor(
        [
            _fit_branch_length(parent, child, _prob_of_branch_simple_zapped)
            for parent, child in zip(pcp_df["parent_h"], pcp_df["child_h"])
        ]
    )

    no_mh_burrito = make_dasm_burrito(None, pcp_df)
    optimization_kwargs = {"learning_rate": 0.01, "optimization_tol": 1e-3}
    lengths = no_mh_burrito.serial_find_optimal_branch_lengths(
        no_mh_burrito.train_dataset, **optimization_kwargs
    )
    if not torch.allclose(
        lengths.double(), validated_no_mh_lengths.double(), atol=1e-4
    ):
        print("Branch lengths don't match:")
        print("burrito lengths:  ", lengths.double())
        print("validated lengths:", validated_no_mh_lengths.double())
        assert False

    # There should be four distinct values here, and it seems that there are.
    print(
        Counter(
            no_mh_burrito.train_dataset.log_neutral_codon_probss[0].numpy().flatten()
        )
    )
    null_mh_model = HitClassModel()
    null_mh_burrito = make_dasm_burrito(null_mh_model, pcp_df)
    null_mh_lengths = null_mh_burrito.serial_find_optimal_branch_lengths(
        null_mh_burrito.train_dataset, **optimization_kwargs
    )
    assert torch.allclose(lengths, null_mh_lengths)

    mh_model = HitClassModel()
    mh_model.values = torch.nn.Parameter(torch.tensor([1.0, 2.0, 5]).log())
    mh_burrito = make_dasm_burrito(mh_model, pcp_df)
    mh_lengths = mh_burrito.serial_find_optimal_branch_lengths(
        mh_burrito.train_dataset, **optimization_kwargs
    )
    print(lengths)
    print(mh_lengths)
    # Strange things: Applying the multihit model above decreases the
    # branch length for all sequences. I'd expect it to increase slightly for
    # the sequence with only one mutation.
    # Finally, I compute ML branch lengths of [0.1178, 0.2513, 0.405, 0.588]
    # without the multihit correction. These don't quite match the
    # branch lengths computed by the uncorrected model.
    assert not torch.allclose(lengths, mh_lengths)


def test_multihit_branch_lengths_dnsm():
    pcp_df = make_test_pcp_df()

    print(list(pcp_df["nt_rates_h"]))
    print(list(pcp_df["nt_csps_h"]))

    no_mh_burrito = make_dnsm_burrito(None, pcp_df)
    optimization_kwargs = {"learning_rate": 0.01, "optimization_tol": 1e-3}
    lengths = no_mh_burrito.serial_find_optimal_branch_lengths(
        no_mh_burrito.train_dataset, **optimization_kwargs
    )

    null_mh_model = HitClassModel()
    null_mh_burrito = make_dnsm_burrito(null_mh_model, pcp_df)
    null_mh_lengths = null_mh_burrito.serial_find_optimal_branch_lengths(
        null_mh_burrito.train_dataset, **optimization_kwargs
    )
    assert torch.allclose(lengths, null_mh_lengths)

    mh_model = HitClassModel()
    mh_model.values = torch.nn.Parameter(torch.tensor([1.0, 2.0, 5]).log())
    mh_burrito = make_dnsm_burrito(mh_model, pcp_df)
    mh_lengths = mh_burrito.serial_find_optimal_branch_lengths(
        mh_burrito.train_dataset, **optimization_kwargs
    )
    print(lengths)
    print(mh_lengths)
    assert not torch.allclose(lengths, mh_lengths)
