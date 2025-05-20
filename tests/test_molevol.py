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
from netam.framework import add_shm_model_outputs_to_pcp_df, codon_probs_of_parent_seq
from netam.hit_class import parent_specific_hit_classes
from netam.common import clamp_probability, clamp_log_probability, clamp_probability_above

from test_dnsm import dnsm_burrito

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


def test_build_codon_mutsel(pcp_df, dnsm_burrito):
    # There are two ways of computing codon probabilities. Let's make sure
    # they're the same:
    neutral_crepe = pretrained.load("ThriftyHumV0.2-59")
    pcp_df = add_shm_model_outputs_to_pcp_df(
        pcp_df.copy(),
        neutral_crepe,
    )
    multihit_model = pretrained.load_multihit(DEFAULT_MULTIHIT_MODEL)

    branch_length = 0.5
    for seq, nt_rates, nt_csps in zip(pcp_df["parent_h"], pcp_df["nt_rates_h"], pcp_df["nt_csps_h"]):
        parent_idxs = sequences.nt_idx_tensor_of_str(seq)
        aa_parent_idxs = sequences.aa_idx_tensor_of_str(
            translate_sequence(seq)
        )
        aa_seq_len = len(seq) // 3
        codon_parent_idxs = sequences.codon_idx_tensor_of_str_ambig(seq)
        hit_classes = parent_specific_hit_classes(
            parent_idxs.reshape(-1, 3),
        )
        flat_hit_classes = molevol.flatten_codons(hit_classes)

        aa_mask = torch.full_like(aa_parent_idxs, True).bool()
        # sel_matrix = torch.ones((aa_seq_len, 20))
        sel_matrix = dnsm_burrito.build_selection_matrix_from_parent_aa(aa_parent_idxs, aa_mask)
        # neutral_sel_matrix[torch.arange(aa_seq_len), aa_parent_idxs]

        # First way:
        nt_mut_probs = 1.0 - torch.exp(-branch_length * nt_rates)
        codon_mutsel, _ = molevol.build_codon_mutsel(
            parent_idxs.reshape(-1, 3),
            nt_mut_probs.reshape(-1, 3),
            nt_csps.reshape(-1, 3, 4),
            sel_matrix,
            multihit_model=multihit_model,
        )
        log_codon_mutsel = clamp_probability(codon_mutsel).log()
        flat_log_codon_mutsel = molevol.flatten_codons(log_codon_mutsel)

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
        if not torch.allclose(adjusted_codon_probs, flat_log_codon_mutsel):
            diff_mask = ~torch.isclose(adjusted_codon_probs, flat_log_codon_mutsel)
            print(flat_hit_classes[diff_mask])
            print((adjusted_codon_probs - flat_log_codon_mutsel)[diff_mask])
            print(adjusted_codon_probs[diff_mask])
            print(flat_log_codon_mutsel[diff_mask])
            assert False

        # Now let's compare to the simulation probs:
        sim_probs = clamp_probability(codon_probs_of_parent_seq(
            dnsm_burrito.to_crepe(),
            (seq, ""),
            branch_length,
            neutral_crepe=neutral_crepe,
            multihit_model=multihit_model,
        )[0]).log()

        if not torch.allclose(adjusted_codon_probs, sim_probs):
            diff_mask = ~torch.isclose(adjusted_codon_probs, sim_probs)
            print(flat_hit_classes[diff_mask].detach().numpy())
            print((adjusted_codon_probs - sim_probs)[diff_mask].detach().numpy())
            print(adjusted_codon_probs[diff_mask].detach().numpy())
            print(sim_probs[diff_mask].detach().numpy())
            assert False
