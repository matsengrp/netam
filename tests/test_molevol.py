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
ex_sub_probs = torch.tensor(
    [[0.0, 0.3, 0.5, 0.2], [0.4, 0.0, 0.1, 0.5], [0.2, 0.3, 0.0, 0.5]]
)
ex_parent_codon_idxs = nt_idx_tensor_of_str("ACG")
parent_nt_seq = "CAGGTGCAGCTGGTGGAG"  # QVQLVE
weights_path = "data/shmple_weights/my_shmoof"


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
