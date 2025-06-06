import torch
import pytest

import netam.molevol as molevol
from netam import pretrained
import netam.sequences as sequences
from netam.models import DEFAULT_MULTIHIT_MODEL

from netam.sequences import (
    nt_idx_tensor_of_str,
    translate_sequence,
    AA_STR_SORTED,
    CODONS,
    NT_STR_SORTED,
)
from netam.common import clamp_probability
from netam.dasm import (
    DASMBurrito,
    DASMDataset,
)
from netam.dnsm import (
    DNSMBurrito,
    DNSMDataset,
)

_dxsm_classes_of_name = {
    "dasm": (DASMDataset, DASMBurrito),
    "dnsm": (DNSMDataset, DNSMBurrito),
}

# These happen to be the same as some examples in test_models.py but that's fine.
# If it was important that they were shared, we should put them in a conftest.py.
ex_mut_probs = torch.tensor([0.01, 0.02, 0.03])
ex_csps = torch.tensor(
    [[0.0, 0.3, 0.5, 0.2], [0.4, 0.0, 0.1, 0.5], [0.2, 0.3, 0.0, 0.5]]
)

ex_parent_codon_idxs = nt_idx_tensor_of_str("ACG")
parent_nt_seq = "CAGGTGCAGCTGGTGGAG"  # QVQLVE


def test_aaprobs_of_parent_scaled_rates_and_csps():

    def old_aaprobs_of_parent_scaled_rates_and_csps(
        parent_idxs: torch.Tensor, scaled_rates: torch.Tensor, csps: torch.Tensor
    ) -> torch.Tensor:
        """Calculate per-site amino acid probabilities from per-site nucleotide rates
        and substitution probabilities.

        Args:
            parent_idxs (torch.Tensor): Parent nucleotide indices. Shape should be (site_count,).
            scaled_rates (torch.Tensor): Poisson rates of mutation per site, scaled by branch length.
                                         Shape should be (site_count,).
            csps (torch.Tensor): Substitution probabilities per site: a 2D
                                      tensor with shape (site_count, 4).

        Returns:
            torch.Tensor: A 2D tensor with rows corresponding to sites and columns
                          corresponding to amino acids.
        """
        # Calculate the probability of at least one mutation at each site.
        mut_probs = 1.0 - torch.exp(-scaled_rates)

        # Reshape the inputs to include a codon dimension.
        parent_codon_idxs = molevol.reshape_for_codons(parent_idxs)
        codon_mut_probs = molevol.reshape_for_codons(mut_probs)
        codon_csps = molevol.reshape_for_codons(csps)

        # Vectorized calculation of amino acid probabilities.
        return molevol.aaprob_of_mut_and_sub(
            parent_codon_idxs, codon_mut_probs, codon_csps
        )

    assert torch.allclose(
        old_aaprobs_of_parent_scaled_rates_and_csps(
            ex_parent_codon_idxs, ex_mut_probs, ex_csps
        ),
        molevol.aaprobs_of_parent_scaled_rates_and_csps(
            ex_parent_codon_idxs, ex_mut_probs, ex_csps
        ),
    )


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
        ex_csps.unsqueeze(0),
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
        ex_csps.unsqueeze(0),
    ).squeeze()

    assert torch.allclose(correct_tensor, computed_tensor)


def test_check_csps():
    parent_idxs = nt_idx_tensor_of_str("AC")
    csp = torch.tensor([[0.0, 0.375, 0.5, 0.125], [0.125, 0.0, 0.375, 0.5]])
    molevol.check_csps(parent_idxs, csp)

    not_csp = torch.tensor([[0.2, 0.3, 0.4, 0.1], [0.1, 0.2, 0.3, 0.4]])
    with pytest.raises(AssertionError):
        molevol.check_csps(parent_idxs, not_csp)


def iterative_aaprob_of_mut_and_sub(parent_codon, mut_probs, csps):
    """Original version of codon_to_aa_probabilities, used for testing."""
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
                child_prob *= csps[isite][NT_STR_SORTED.index(child_codon[isite])]

        aa_probs[aa] += child_prob

    # need renormalization factor so that amino acid probabilities sum to 1,
    # since probabilities to STOP codon are dropped
    psum = sum(aa_probs.values())

    return torch.tensor([aa_probs[aa] / psum for aa in AA_STR_SORTED])


def test_aaprob_of_mut_and_sub():
    crepe = pretrained.load("ThriftyHumV0.2-45")
    [rates], [subs] = crepe([parent_nt_seq])
    mut_probs = 1.0 - torch.exp(-rates.squeeze().clone().detach())
    parent_codon = parent_nt_seq[0:3]
    parent_codon_idxs = nt_idx_tensor_of_str(parent_codon)
    codon_mut_probs = mut_probs[0:3]
    codon_subs = subs.clone().detach()[0:3]

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


def _check_non_stop_neutral_aa_mut_probs(
    nt_parent_idxs,
    nt_rates,
    nt_csps,
    branch_length,
    multihit_model=None,
):

    mut_probs = 1.0 - torch.exp(-branch_length * nt_rates)
    parent_codon_idxs = nt_parent_idxs.reshape(-1, 3)
    flat_parent_codon_idxs = sequences.flatten_codon_idxs(parent_codon_idxs)
    mut_matrices = molevol.build_mutation_matrices(
        parent_codon_idxs,
        mut_probs.reshape(-1, 3),
        nt_csps.reshape(-1, 3, 4),
    )
    codon_probs = molevol.codon_probs_of_mutation_matrices(mut_matrices)

    if multihit_model is not None:
        codon_probs = multihit_model.forward(parent_codon_idxs, codon_probs)

    flat_codon_probs = molevol.zero_stop_codon_probs(
        molevol.flatten_codons(codon_probs)
    )
    flat_codon_probs = molevol.set_parent_codon_prob(
        flat_codon_probs, flat_parent_codon_idxs
    )
    aa_probs = flat_codon_probs @ molevol.CODON_AA_INDICATOR_MATRIX

    parent_aa_idxs = molevol.aa_idxs_of_codon_idxs(parent_codon_idxs)
    aa_probs[torch.arange(len(parent_aa_idxs)), parent_aa_idxs] = 0.0
    sums = aa_probs.sum(dim=-1)
    return clamp_probability(sums)


def test_non_stop_neutral_aa_mut_probs(pcp_df):
    branch_length = 0.05
    for multihit_model in [None, pretrained.load_multihit(DEFAULT_MULTIHIT_MODEL)]:
        for seq, rates, csps in zip(
            pcp_df["parent_heavy"], pcp_df["nt_rates_heavy"], pcp_df["nt_csps_heavy"]
        ):
            neutral_aa_probs = molevol.non_stop_neutral_aa_mut_probs(
                nt_idx_tensor_of_str(seq),
                rates,
                csps,
                branch_length,
                multihit_model=multihit_model,
            )
            check_neutral_aa_probs = _check_non_stop_neutral_aa_mut_probs(
                nt_idx_tensor_of_str(seq),
                rates,
                csps,
                branch_length,
                multihit_model=multihit_model,
            )
            assert torch.allclose(neutral_aa_probs, check_neutral_aa_probs)
