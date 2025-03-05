import os
from collections import Counter

import netam.multihit as multihit
import netam.framework as framework
import netam.hit_class as hit_class
from netam.molevol import (
    codon_probs_of_parent_scaled_nt_rates_and_csps,
    reshape_for_codons,
)
from netam import pretrained
from netam.sequences import nt_idx_tensor_of_str, MAX_KNOWN_TOKEN_COUNT
from netam.models import SingleValueBinarySelectionModel, HitClassModel
from netam.dasm import DASMDataset, DASMBurrito
from netam.dnsm import DNSMDataset, DNSMBurrito
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
    dataset = DASMDataset.of_pcp_df(pcp_df, MAX_KNOWN_TOKEN_COUNT, multihit_model=multihit_model)
    model = SingleValueBinarySelectionModel(
        output_dim=20,
        known_token_count=MAX_KNOWN_TOKEN_COUNT,
    )
    model.single_value = torch.nn.Parameter(torch.tensor(0.0))
    burrito = DASMBurrito(dataset, dataset, model, batch_size=2, learning_rate=0.001, min_learning_rate=0.0001)
    return burrito

def make_dnsm_burrito(multihit_model, pcp_df):
    dataset = DNSMDataset.of_pcp_df(pcp_df, MAX_KNOWN_TOKEN_COUNT, multihit_model=multihit_model)
    model = SingleValueBinarySelectionModel()
    model.single_value = torch.nn.Parameter(torch.tensor(0.0))
    burrito = DNSMBurrito(dataset, dataset, model, batch_size=2, learning_rate=0.001, min_learning_rate=0.0001)
    return burrito


def test_multihit_branch_lengths_dasm():
    # Here we have a hit class 0 codon and a hit class 1 codon in each pair,
    # and one pair each with a 0, 1, , 2, and 3 hit class codon.
    parent_seq = "ATGTACTTA"
    child_seqs = ["ATGTACTCA", "ATGTATTCA", "ATGTTGTCA", "ATGGTGTCA"]
    df = pd.DataFrame({"parent": [parent_seq] * 4, "child": child_seqs, "v_gene": ["IGHV1-39*01"] * 4})
    for child in child_seqs:
        print(sum(c1 != c2 for c1, c2 in zip(parent_seq, child)))
    # df = pd.DataFrame({"parent": ["ATGTAC"] * 3, "child": ["ATGTAT", "ATGTTG", "ATGGTG"], "v_gene": ["IGHV1-39*01"] * 3})
    pcp_df = framework.standardize_heavy_light_columns(df)
    pcp_df = framework.add_shm_model_outputs_to_pcp_df(pcp_df, pretrained.load("ThriftyHumV0.2-45"))
    # Neutralize neutral rates
    pcp_df["nt_rates_h"] = [torch.tensor([1.0] * 9)] * len(pcp_df)
    nt_csps = list(pcp_df["nt_csps_h"])
    for i in range(len(nt_csps)):
        val = nt_csps[i]
        val[val > 0.0] = 1.0 / 3.0
    pcp_df["nt_csps_h"] = nt_csps

    print(list(pcp_df["nt_rates_h"]))
    print(list(pcp_df["nt_csps_h"]))

    no_mh_burrito = make_dasm_burrito(None, pcp_df)
    optimization_kwargs = {"learning_rate": 0.01, "optimization_tol": 1e-3}
    lengths = no_mh_burrito.serial_find_optimal_branch_lengths(no_mh_burrito.train_dataset, **optimization_kwargs)

    # There should be four distinct values here, and it seems that there are.
    print(Counter(no_mh_burrito.train_dataset.log_neutral_codon_probss[0].numpy().flatten()))
    null_mh_model = HitClassModel()
    null_mh_burrito = make_dasm_burrito(null_mh_model, pcp_df)
    null_mh_lengths = null_mh_burrito.serial_find_optimal_branch_lengths(null_mh_burrito.train_dataset, **optimization_kwargs)
    assert torch.allclose(lengths, null_mh_lengths)

    mh_model = HitClassModel()
    mh_model.values = torch.nn.Parameter(torch.tensor([1.0,  2.0,  5]).log())
    mh_burrito = make_dasm_burrito(mh_model, pcp_df)
    mh_lengths = mh_burrito.serial_find_optimal_branch_lengths(mh_burrito.train_dataset, **optimization_kwargs)
    print(lengths)
    print(mh_lengths)
    # Strange things: Applying the multihit model above decreases the
    # branch length for all sequences. I'd expect it to increase slightly for
    # the sequence with only one mutation.
    # When I look at the neutral codon probs, it seems that there are more
    # than four distinct values, and I would expect there to be exactly four
    # (one for each hit class).
    # Finally, I compute ML branch lengths of [0.1178, 0.2513, 0.405, 0.588]
    # without the multihit correction. These don't quite match the
    # branch lengths computed by the uncorrected model.
    assert not torch.allclose(lengths, mh_lengths)


def test_multihit_branch_lengths_dnsm():
    # Here we have a hit class 0 codon and a hit class 1 codon in each pair,
    # and one pair each with a 0, 1, , 2, and 3 hit class codon.
    parent_seq = "ATGTACTTA"
    child_seqs = ["ATGTACTCA", "ATGTATTCA", "ATGTTGTCA", "ATGGTGTCA"]
    df = pd.DataFrame({"parent": [parent_seq] * 4, "child": child_seqs, "v_gene": ["IGHV1-39*01"] * 4})
    for child in child_seqs:
        print(sum(c1 != c2 for c1, c2 in zip(parent_seq, child)))
    # df = pd.DataFrame({"parent": ["ATGTAC"] * 3, "child": ["ATGTAT", "ATGTTG", "ATGGTG"], "v_gene": ["IGHV1-39*01"] * 3})
    pcp_df = framework.standardize_heavy_light_columns(df)
    pcp_df = framework.add_shm_model_outputs_to_pcp_df(pcp_df, pretrained.load("ThriftyHumV0.2-45"))
    # Neutralize neutral rates
    pcp_df["nt_rates_h"] = [torch.tensor([1.0] * 9)] * len(pcp_df)
    nt_csps = list(pcp_df["nt_csps_h"])
    for i in range(len(nt_csps)):
        val = nt_csps[i]
        val[val > 0.0] = 1.0 / 3.0
    pcp_df["nt_csps_h"] = nt_csps

    print(list(pcp_df["nt_rates_h"]))
    print(list(pcp_df["nt_csps_h"]))

    no_mh_burrito = make_dnsm_burrito(None, pcp_df)
    optimization_kwargs = {"learning_rate": 0.01, "optimization_tol": 1e-3}
    lengths = no_mh_burrito.serial_find_optimal_branch_lengths(no_mh_burrito.train_dataset, **optimization_kwargs)

    null_mh_model = HitClassModel()
    null_mh_burrito = make_dnsm_burrito(null_mh_model, pcp_df)
    null_mh_lengths = null_mh_burrito.serial_find_optimal_branch_lengths(null_mh_burrito.train_dataset, **optimization_kwargs)
    assert torch.allclose(lengths, null_mh_lengths)

    mh_model = HitClassModel()
    mh_model.values = torch.nn.Parameter(torch.tensor([1.0,  2.0,  5]).log())
    mh_burrito = make_dnsm_burrito(mh_model, pcp_df)
    mh_lengths = mh_burrito.serial_find_optimal_branch_lengths(mh_burrito.train_dataset, **optimization_kwargs)
    print(lengths)
    print(mh_lengths)
    assert not torch.allclose(lengths, mh_lengths)
