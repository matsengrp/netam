import torch
import netam.molevol as molevol
from netam import framework

from netam.sequences import (
    nt_idx_tensor_of_str,
    translate_sequence,
    AA_STR_SORTED,
    CODONS,
    NT_STR_SORTED,
)

# These happen to be the same as some examples in test_models.py but that's fine.
# If it was important that they were shared, we should put them in a conftest.py.
ex_mut_probs = torch.tensor([0.01, 0.02, 0.03])
ex_scaled_rates = torch.tensor([0.01, 0.001, 0.005])
ex_sub_probs = torch.tensor(
    [[0.0, 0.3, 0.5, 0.2], [0.4, 0.0, 0.1, 0.5], [0.2, 0.3, 0.0, 0.5]]
)
# This is an example, and the correct output for test_codon_probs_of_parent_scaled_rates_and_sub_probs
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
parent_nt_seq = "CAGGTGCAGCTGGTGGAG"  # QVQLVE
weights_path = "data/shmple_weights/my_shmoof"


def test_codon_probs_of_parent_scaled_rates_and_sub_probs():
    computed_tensor = molevol.codon_probs_of_parent_scaled_rates_and_sub_probs(
        ex_parent_codon_idxs, ex_scaled_rates, ex_sub_probs
    )
    correct_tensor = ex_codon_probs
    assert torch.allclose(correct_tensor, computed_tensor)
    assert torch.allclose(
        computed_tensor.sum(dim=(1, 2, 3)), torch.ones(computed_tensor.shape[0])
    )


def test_multihit_adjustment():
    hit_class_factors = torch.tensor([-0.1, 1, 2.3])
    # We'll verify that aggregating by hit class then adjusting is the same as adjusting then aggregating by hit class.
    codon_idxs = molevol.reshape_for_codons(ex_parent_codon_idxs)
    adjusted_codon_probs = molevol.apply_multihit_adjustment(
        codon_idxs, ex_codon_probs.log(), hit_class_factors
    ).exp()
    aggregate_last = molevol.hit_class_probs_tensor(codon_idxs, adjusted_codon_probs)

    uncorrected_hc_log_probs = molevol.hit_class_probs_tensor(
        codon_idxs, ex_codon_probs
    ).log()

    corrections = torch.cat([torch.tensor([0.0]), hit_class_factors])
    # we'll use the corrections to adjust the uncorrected hit class probs
    adjustments = corrections[
        torch.arange(4).unsqueeze(0).tile((uncorrected_hc_log_probs.shape[0], 1))
    ]
    uncorrected_hc_log_probs += adjustments
    aggregate_first = torch.softmax(uncorrected_hc_log_probs, dim=1)
    assert torch.allclose(aggregate_first, aggregate_last)


def test_build_mutation_matrix():
    correct_tensor = torch.tensor(
        [
            # probability of mutation to each nucleotide (first entry in the first row
            # is probability of no mutation)
            [0.99, 0.003, 0.005, 0.002],
            [0.008, 0.98, 0.002, 0.01],
            [0.006, 0.009, 0.97, 0.015],
        ]
    )

    computed_tensor = molevol.build_mutation_matrices(
        ex_parent_codon_idxs.unsqueeze(0),
        ex_mut_probs.unsqueeze(0),
        ex_sub_probs.unsqueeze(0),
    ).squeeze()

    assert torch.allclose(correct_tensor, computed_tensor)


def test_neutral_aa_mut_probs():
    # This is the probability of a mutation to a codon that translates to the
    # same. In this case, ACG is the codon, and it's fourfold degenerate. Thus
    # we just multiply the probability of A and C staying the same from the
    # correct_tensor just above.
    correct_tensor = torch.tensor([1 - 0.99 * 0.98])

    computed_tensor = molevol.neutral_aa_mut_probs(
        ex_parent_codon_idxs.unsqueeze(0),
        ex_mut_probs.unsqueeze(0),
        ex_sub_probs.unsqueeze(0),
    ).squeeze()

    assert torch.allclose(correct_tensor, computed_tensor)


def test_normalize_sub_probs():
    parent_idxs = nt_idx_tensor_of_str("AC")
    sub_probs = torch.tensor([[0.2, 0.3, 0.4, 0.1], [0.1, 0.2, 0.3, 0.4]])

    expected_normalized = torch.tensor(
        [[0.0, 0.375, 0.5, 0.125], [0.125, 0.0, 0.375, 0.5]]
    )
    normalized_sub_probs = molevol.normalize_sub_probs(parent_idxs, sub_probs)

    assert normalized_sub_probs.shape == (2, 4), "Result has incorrect shape"
    assert torch.allclose(
        normalized_sub_probs, expected_normalized
    ), "Unexpected normalized values"


def iterative_aaprob_of_mut_and_sub(parent_codon, mut_probs, sub_probs):
    """
    Original version of codon_to_aa_probabilities, used for testing.
    """
    aa_probs = {}
    for aa in AA_STR_SORTED:
        aa_probs[aa] = 0.0

    # iterate through all possible child codons
    for child_codon in CODONS:
        try:
            aa = translate_sequence(child_codon)
        except ValueError:  # check for STOP codon
            continue

        # iterate through codon sites and compute total probability of potential child codon
        child_prob = 1.0
        for isite in range(3):
            if parent_codon[isite] == child_codon[isite]:
                child_prob *= 1.0 - mut_probs[isite]
            else:
                child_prob *= mut_probs[isite]
                child_prob *= sub_probs[isite][NT_STR_SORTED.index(child_codon[isite])]

        aa_probs[aa] += child_prob

    # need renormalization factor so that amino acid probabilities sum to 1,
    # since probabilities to STOP codon are dropped
    psum = sum(aa_probs.values())

    return torch.tensor([aa_probs[aa] / psum for aa in AA_STR_SORTED])


def test_aaprob_of_mut_and_sub():
    crepe_path = "data/cnn_joi_sml-shmoof_small"
    crepe = framework.load_crepe(crepe_path)
    [rates], [subs] = crepe([parent_nt_seq])
    mut_probs = 1.0 - torch.exp(-torch.tensor(rates.squeeze()))
    parent_codon = parent_nt_seq[0:3]
    parent_codon_idxs = nt_idx_tensor_of_str(parent_codon)
    codon_mut_probs = mut_probs[0:3]
    codon_subs = torch.tensor(subs[0:3])

    iterative_result = iterative_aaprob_of_mut_and_sub(
        parent_codon, codon_mut_probs, codon_subs
    )

    parent_codon_idxs = parent_codon_idxs.unsqueeze(0)
    codon_mut_probs = codon_mut_probs.unsqueeze(0)
    codon_subs = codon_subs.unsqueeze(0)

    assert torch.allclose(
        iterative_result,
        molevol.aaprob_of_mut_and_sub(
            parent_codon_idxs,
            codon_mut_probs,
            codon_subs,
        ).squeeze(),
    )
