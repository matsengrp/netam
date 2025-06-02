"""Free functions for molecular evolution computation.

CSP means conditional substitution probability. CSPs are the probabilities of alternate
states conditioned on there being a substitution.
"""

import numpy as np
from scipy import optimize

import torch
from torch import Tensor
from warnings import warn

from netam.codon_table import (
    CODON_AA_INDICATOR_MATRIX,
    STOP_CODON_ZAPPER,
    aa_idxs_of_codon_idxs,
)

import netam.sequences as sequences
from netam.common import clamp_probability, clamp_probability_above_only


def check_csps(parent_idxs: Tensor, csps: Tensor) -> Tensor:
    """Make sure that the CSPs are valid, i.e. that they are a probability distribution
    and the parent state is zero.

    Args:
        parent_idxs (torch.Tensor): The parent sequence indices.
        sub_probs (torch.Tensor): A 2D PyTorch tensor representing substitution
            probabilities. Rows correspond to sites, and columns correspond
            to states (e.g. nucleotides).
    """

    if len(parent_idxs) == 0:
        raise ValueError("Parent indices must not be empty")
    # Assert that sub_probs are within the range [0, 1] modulo rounding error
    assert torch.all(csps >= -1e-6), "Substitution probabilities must be non-negative"
    assert torch.all(
        csps <= 1 + 1e-6
    ), "Substitution probabilities must be less than or equal to 1"

    # Create an array of row indices that matches the shape of `parent_idxs`.
    row_indices = torch.arange(len(parent_idxs))

    # Assert that the parent entry has a substitution probability of nearly 0.
    assert torch.all(
        csps[row_indices, parent_idxs] < 1e-6
    ), "Parent entry must have a substitution probability of nearly 0"
    assert torch.allclose(
        csps[: len(parent_idxs)].sum(dim=1), torch.ones(len(parent_idxs))
    ), "Substitution probabilities must sum to 1"


def build_mutation_matrices(
    parent_codon_idxs: Tensor, codon_mut_probs: Tensor, codon_csps: Tensor
) -> Tensor:
    """Generate a sequence of 3x4 mutation matrices for parent codons along a sequence.

    Given indices for parent codons, mutation probabilities, and substitution
    probabilities for each parent codon along the sequence, this function
    constructs a sequence of 3x4 matrices. Each matrix in the sequence
    represents the mutation probabilities for each nucleotide position in a
    parent codon. The ijkth entry of the resulting tensor corresponds to the
    probability of the jth nucleotide in the ith parent codon mutating to the
    kth nucleotide (in indices).

    Args:
        parent_codon_idxs (torch.Tensor): 2D tensor with each row containing indices representing
            the parent codon's nucleotides at each site along the sequence.
            Shape should be (codon_count, 3).
        codon_mut_probs (torch.Tensor): 2D tensor representing the mutation probabilities for each site in the codon,
            for each codon along the sequence. Shape should be (codon_count, 3).
        codon_csps (torch.Tensor): 3D tensor representing conditional substitution probabilities for each NT site of each codon along the
            sequence. Shape should be (codon_count, 3, 4).

    Returns:
        torch.Tensor: A 4D tensor with shape (codon_count, 3, 4) where the ijkth entry is the mutation probability
            of the jth position in the ith parent codon mutating to the kth nucleotide.
    """

    codon_count = parent_codon_idxs.shape[0]
    assert parent_codon_idxs.shape[1] == 3, "Each parent codon must be of length 3"

    result_matrices = torch.empty((codon_count, 3, 4))

    # Create a mask with the shape (codon_count, 3, 4) to identify where
    # nucleotides in the parent codon are the same as the nucleotide positions
    # in the new codon. Each row in the third dimension contains a boolean
    # value, which is True if the nucleotide position matches the parent codon
    # nucleotide. How it works: None adds one more dimension to the tensor, so
    # that the shape of the tensor is (codon_count, 3, 1) instead of
    # (codon_count, 3). Then broadcasting automatically expands dimensions where
    # needed. So the arange(4) tensor is automatically expanded to match the
    # (codon_count, 3, 1) shape by implicitly turning it into a (1, 1, 4) shape
    # tensor, where it is then broadcasted to the shape (codon_count, 3, 4) to
    # match the shape of parent_codon_idxs[:, :, None] for equality
    # testing.
    mask_same_nt = torch.arange(4) == parent_codon_idxs[:, :, None]

    # Find the multi-dimensional indices where the nucleotide in the parent
    # codon is the same as the nucleotide in the mutation outcome (i.e., no
    # mutation occurs).
    same_nt_indices = torch.nonzero(mask_same_nt)

    # Using the multi-dimensional indices obtained from the boolean mask, update
    # the mutation probability in result_matrices to be "1.0 -
    # mutation_probability" at these specific positions. This captures the
    # probability of a given nucleotide not mutating.
    result_matrices[
        same_nt_indices[:, 0], same_nt_indices[:, 1], same_nt_indices[:, 2]
    ] = (1.0 - codon_mut_probs[same_nt_indices[:, 0], same_nt_indices[:, 1]])

    # Assign values where the nucleotide is different via broadcasting.
    mask_diff_nt = ~mask_same_nt
    result_matrices[mask_diff_nt] = (codon_mut_probs[:, :, None] * codon_csps)[
        mask_diff_nt
    ]

    return result_matrices


def codon_probs_of_mutation_matrices(mut_matrix: Tensor) -> Tensor:
    """Compute the probability tensor for mutating to the codon ijk along the entire
    sequence.

    Args:
    mut_matrix (torch.Tensor): A 3D tensor representing the mutation matrix for the entire sequence.
                               The shape should be (n_sites, 3, 4), where n_sites is the number of sites,
                               3 is the number of positions in a codon, and 4 is the number of nucleotides.

    Returns:
    torch.Tensor: A 4D tensor where the first axis represents different sites in the sequence and
                  the ijk-th entry of the remaining 3D tensor is the probability of mutating to the codon ijk.
    """
    assert (
        mut_matrix.shape[1] == 3
    ), "The second dimension of the input mut_matrix should be 3 to represent the 3 positions in a codon"
    assert (
        mut_matrix.shape[2] == 4
    ), "The last dimension of the input mut_matrix should be 4 to represent the 4 nucleotides"

    # The key to understanding how this works is that when these tensors are
    # multiplied, PyTorch broadcasts them into a common shape (n_sites, 4, 4, 4),
    # performing element-wise multiplication for each slice along the first axis
    # (i.e., for each site).
    return (
        mut_matrix[:, 0, :, None, None]
        * mut_matrix[:, 1, None, :, None]
        * mut_matrix[:, 2, None, None, :]
    )


def aaprobs_of_codon_probs(codon_probs: Tensor) -> Tensor:
    """Compute the probability of each amino acid from the probability of each codon,
    for each parent codon along the sequence.

    Args:
    codon_probs (torch.Tensor): A 4D tensor representing the probability of mutating
                                to each codon for each parent codon along the sequence.
                                Shape should be (codon_count, 4, 4, 4).

    Returns:
    torch.Tensor: A 2D tensor with shape (codon_count, 20) where the ij-th entry is the probability
                  of mutating to the amino acid j from the codon i for each parent codon along the sequence.
    """
    codon_count = codon_probs.shape[0]

    # Reshape such that we merge the last three dimensions into a single dimension while keeping
    # the `codon_count` dimension intact. This prepares the tensor for matrix multiplication.
    reshaped_probs = codon_probs.reshape(codon_count, -1)

    # Perform matrix multiplication to get unnormalized amino acid probabilities.
    aaprobs = torch.matmul(reshaped_probs, CODON_AA_INDICATOR_MATRIX)

    # Normalize probabilities along the amino acid dimension.
    row_sums = aaprobs.sum(dim=1, keepdim=True)
    aaprobs /= row_sums

    return aaprobs


def aaprob_of_mut_and_sub(
    parent_codon_idxs: Tensor, codon_mut_probs: Tensor, codon_csps: Tensor
) -> Tensor:
    """For a sequence of parent codons and given nucleotide mutability and substitution
    probabilities, compute the probability of a substitution to each amino acid for each
    codon along the sequence.

    This function actually isn't used anymore, but there is a good test for it, which
    tests other functions, so we keep it.

    Stop codons don't appear as part of this calculation.

    Args:
    parent_codon_idxs (torch.Tensor): A 2D tensor where each row contains indices representing
                                      the parent codon's nucleotides at each site along the sequence.
                                      Shape should be (codon_count, 3).
    codon_mut_probs (torch.Tensor): A 2D tensor representing the mutation probabilities for each site in the codon,
                              for each codon along the sequence. Shape should be (codon_count, 3).
    codon_csps (torch.Tensor): A 3D tensor representing conditional substitution probabilities for each NT site of each codon along the
                                sequence.
                                Shape should be (codon_count, 3, 4).

    Returns:
    torch.Tensor: A 2D tensor with shape (codon_count, 20) where the ij-th entry is the probability
                  of mutating to the amino acid j from the codon i for each parent codon along the sequence.
    """
    mut_matrices = build_mutation_matrices(
        parent_codon_idxs, codon_mut_probs, codon_csps
    )
    codon_probs = codon_probs_of_mutation_matrices(mut_matrices)
    return aaprobs_of_codon_probs(codon_probs)


def flatten_codons(array: Tensor) -> Tensor:
    """Reshape a tensor from (..., 4, 4, 4) to (..., 64)."""
    shape = array.shape
    assert shape[-3:] == (4, 4, 4), "Last three dimensions must be (4, 4, 4)"
    return array.reshape(shape[:-3] + (64,))


def unflatten_codons(array: Tensor) -> Tensor:
    """Reshape a tensor from (..., 64) to (..., 4, 4, 4)."""
    shape = array.shape
    assert shape[-1] == 64, "Last dimension must be 64"
    return array.reshape(shape[:-1] + (4, 4, 4))


def reshape_for_codons(array: Tensor) -> Tensor:
    """Reshape a tensor to add a codon dimension by taking groups of 3 sites.

    Args:
    array (torch.Tensor): Original tensor.

    Returns:
    torch.Tensor: Reshaped tensor with an added codon dimension.
    """
    site_count = array.shape[0]
    assert site_count % 3 == 0, "Site count must be a multiple of 3"
    codon_count = site_count // 3
    return array.reshape(codon_count, 3, *array.shape[1:])


def codon_probs_of_parent_scaled_nt_rates_and_csps(
    parent_idxs: torch.Tensor, scaled_nt_rates: torch.Tensor, nt_csps: torch.Tensor
):
    """Compute the probabilities of mutating to various codons for a parent sequence.

    This uses the same machinery as we use for fitting the DNSM, but we stay on
    the codon level rather than moving to syn/nonsyn changes.

    Args:
        parent_idxs (torch.Tensor): The parent nucleotide sequence encoded as a
            tensor of length Cx3, where C is the number of codons, containing the nt indices of each site.
        scaled_nt_rates (torch.Tensor): Poisson rates of mutation per site, scaled by branch length.
        nt_csps (torch.Tensor): Conditional substitution probabilities per site: a 2D tensor with shape (site_count, 4).

    Returns:
        torch.Tensor: A 4D tensor with shape (codon_count, 4, 4, 4) where the cijk-th entry is the probability
            of the c'th codon mutating to the codon ijk.
    """
    mut_probs = 1.0 - torch.exp(-scaled_nt_rates)
    parent_codon_idxs = reshape_for_codons(parent_idxs)
    codon_mut_probs = reshape_for_codons(mut_probs)
    codon_csps = reshape_for_codons(nt_csps)

    mut_matrices = build_mutation_matrices(
        parent_codon_idxs, codon_mut_probs, codon_csps
    )
    codon_probs = codon_probs_of_mutation_matrices(mut_matrices)

    return codon_probs


def aaprobs_of_parent_scaled_rates_and_csps(
    parent_idxs: Tensor, scaled_nt_rates: Tensor, nt_csps: Tensor
) -> Tensor:
    """Calculate per-site amino acid probabilities from per-site nucleotide rates and
    substitution probabilities.

    Args:
        parent_idxs (torch.Tensor): Parent nucleotide indices. Shape should be (site_count,).
        scaled_nt_rates (torch.Tensor): Poisson rates of mutation per site, scaled by branch length.
                                     Shape should be (site_count,).
        nt_csps (torch.Tensor): Substitution probabilities per site: a 2D
                                  tensor with shape (site_count, 4).

    Returns:
        torch.Tensor: A 2D tensor with rows corresponding to sites and columns
                      corresponding to amino acids.
    """
    return aaprobs_of_codon_probs(
        codon_probs_of_parent_scaled_nt_rates_and_csps(
            parent_idxs, scaled_nt_rates, nt_csps
        )
    )


def build_codon_mutsel(
    parent_codon_idxs: Tensor,
    codon_mut_probs: Tensor,
    codon_csps: Tensor,
    aa_sel_matrices: Tensor,
    multihit_model=None,
) -> Tensor:
    """Build a sequence of codon mutation-selection matrices for codons along a
    sequence.

    These will assign zero for the probability of mutating to a stop codon.

    Args:
        parent_codon_idxs (torch.Tensor): The parent codons for each sequence. Shape: (codon_count, 3)
        codon_mut_probs (torch.Tensor): The mutation probabilities for each site in each codon. Shape: (codon_count, 3)
        codon_csps (torch.Tensor): The substitution probabilities for each site in each codon. Shape: (codon_count, 3, 4)
        aa_sel_matrices (torch.Tensor): The amino-acid selection matrices for each site. Shape: (codon_count, 20)

    Returns:
        torch.Tensor: The probability of mutating to each codon, for each sequence. Shape: (codon_count, 4, 4, 4)
    """
    mut_matrices = build_mutation_matrices(
        parent_codon_idxs, codon_mut_probs, codon_csps
    )
    codon_probs = codon_probs_of_mutation_matrices(mut_matrices)

    if multihit_model is not None:
        codon_probs = multihit_model.forward(parent_codon_idxs, codon_probs)

    # Calculate the codon selection matrix for each sequence via Einstein
    # summation, in which we sum over the repeated indices.
    # So, for each site (s) and codon (c), sum over amino acids (a):
    # codon_sel_matrices[s, c] = sum_a(CODON_AA_INDICATOR_MATRIX[c, a] * aa_sel_matrices[s, a])
    # Resulting shape is (S, C) where S is the number of sites and C is the number of codons.
    # Stop codons don't appear in this sum, so columns for stop codons will be zero.
    codon_sel_matrices = torch.einsum(
        "ca,sa->sc", CODON_AA_INDICATOR_MATRIX, aa_sel_matrices
    )

    # Multiply the codon probabilities by the selection matrices
    codon_mutsel = codon_probs * codon_sel_matrices.view(-1, 4, 4, 4)

    # Clamp the codon_mutsel above by 1: these are probabilities.
    codon_mutsel = codon_mutsel.clamp(max=1.0)

    # Now we need to recalculate the probability of staying in the same codon.
    # In our setup, this is the probability of nothing happening.
    # To calculate this, we zero out the probabilities of mutating to the parent
    # codon...
    codon_count = parent_codon_idxs.shape[0]
    codon_mutsel[(torch.arange(codon_count), *parent_codon_idxs.T)] = 0.0
    # sum together their probabilities...
    sums = codon_mutsel.sum(dim=(1, 2, 3))
    # then set the parent codon probabilities to 1 minus the sum.
    codon_mutsel[(torch.arange(codon_count), *parent_codon_idxs.T)] = 1.0 - sums
    codon_mutsel = codon_mutsel.clamp(min=0.0)

    if sums.max() > 1.0:
        sums_too_big = sums.max()
    else:
        sums_too_big = None

    return codon_mutsel, sums_too_big


def neutral_codon_probs(
    parent_codon_idxs: Tensor,
    codon_mut_probs: Tensor,
    codon_csps: Tensor,
    multihit_model=None,
) -> Tensor:
    """For every site, what is the probability that the site will mutate to every
    alternate codon?

    Args:
        parent_codon_idxs (torch.Tensor): The parent codons for each sequence. Shape: (codon_count, 3)
        codon_mut_probs (torch.Tensor): The mutation probabilities for each site in each codon. Shape: (codon_count, 3)
        codon_csps (torch.Tensor): The substitution probabilities for each site in each codon. Shape: (codon_count, 3, 4)

    Returns:
        torch.Tensor: The probability that each site will change to each codon.
                      Shape: (codon_count, 64)
    """

    mut_matrices = build_mutation_matrices(
        parent_codon_idxs, codon_mut_probs, codon_csps
    )
    codon_probs = codon_probs_of_mutation_matrices(mut_matrices)

    if multihit_model is not None:
        codon_probs = multihit_model.forward(parent_codon_idxs, codon_probs)

    return codon_probs.view(-1, 64)


def neutral_aa_probs(
    parent_codon_idxs: Tensor,
    codon_mut_probs: Tensor,
    codon_csps: Tensor,
    multihit_model=None,
) -> Tensor:
    """For every site, what is the probability that the site will mutate to every
    alternate amino acid?

    Args:
        parent_codon_idxs (torch.Tensor): The parent codons for each sequence. Shape: (codon_count, 3)
        codon_mut_probs (torch.Tensor): The mutation probabilities for each site in each codon. Shape: (codon_count, 3)
        codon_csps (torch.Tensor): The substitution probabilities for each site in each codon. Shape: (codon_count, 3, 4)

    Returns:
        torch.Tensor: The probability that each site will change to each codon.
                      Shape: (codon_count, 20)
    """
    codon_probs = neutral_codon_probs(
        parent_codon_idxs,
        codon_mut_probs,
        codon_csps,
        multihit_model=multihit_model,
    )

    # Get the probability of mutating to each amino acid.
    aa_probs = codon_probs @ CODON_AA_INDICATOR_MATRIX

    return aa_probs


def neutral_codon_probs_of_seq(
    nt_parent: str,
    mask: Tensor,
    nt_rates: Tensor,
    nt_csps: Tensor,
    branch_length: float,
    multihit_model=None,
):
    if len(nt_parent) == 0:
        return torch.empty((0, 64)).float()
    # Note we are replacing all Ns with As, which means that we need to be careful
    # with masking out these positions later. We do this below.
    parent_idxs = sequences.nt_idx_tensor_of_str(nt_parent.replace("N", "A"))
    parent_len = len(nt_parent)

    mut_probs = 1.0 - torch.exp(-branch_length * nt_rates[:parent_len])
    nt_csps = nt_csps[:parent_len, :]
    nt_mask = mask.repeat_interleave(3)[: len(nt_parent)]
    check_csps(parent_idxs[nt_mask], nt_csps[: len(nt_parent)][nt_mask])

    neutral_probs = neutral_codon_probs(
        parent_idxs.reshape(-1, 3),
        mut_probs.reshape(-1, 3),
        nt_csps.reshape(-1, 3, 4),
        multihit_model=multihit_model,
    )

    if not torch.isfinite(neutral_probs).all():
        print("Found a non-finite neutral_codon_prob")
        print(f"nt_parent: {nt_parent}")
        print(f"mask: {mask}")
        print(f"nt_rates: {nt_rates}")
        print(f"nt_csps: {nt_csps}")
        print(f"branch_length: {branch_length}")
        raise ValueError(f"neutral_probs is not finite: {neutral_probs}")
    # Ensure that all values are positive before taking the log later
    neutral_probs = clamp_probability(neutral_probs)

    # Here we zero out masked positions.
    neutral_probs *= mask[: len(neutral_probs), None]
    return neutral_probs


def zero_stop_codon_probs(codon_probs: Tensor):
    """Set stop codon probabilities to zero."""
    return codon_probs * STOP_CODON_ZAPPER.exp()


def adjust_codon_probs_by_aa_selection_factors(
    parent_codon_idxs: Tensor, log_codon_probs: Tensor, log_aa_selection_factors: Tensor
):
    if len(log_codon_probs) == 0:
        return log_codon_probs
    device = parent_codon_idxs.device
    # The aa_codon_indicator_matrix lifts things up from aa land to codon land.
    log_preds = (
        log_codon_probs
        + log_aa_selection_factors @ CODON_AA_INDICATOR_MATRIX.T.to(device)
        + STOP_CODON_ZAPPER.to(device)
    )
    assert torch.isnan(log_preds).sum() == 0

    # Convert to linear space so we can add probabilities.
    preds = torch.exp(log_preds)

    # clamp only above to avoid summing a bunch of small fake values when
    # computing wild type prob
    preds = clamp_probability_above_only(preds)

    preds = set_parent_codon_prob(preds, parent_codon_idxs)

    # We have to clamp the predictions to avoid log(0) issues.
    preds = clamp_probability(preds)

    log_preds = torch.log(preds)

    return log_preds


def mut_probs_of_aa_probs(
    parent_codon_idxs: Tensor,
    aa_probs: Tensor,
) -> Tensor:
    """For every site, what is the probability that the amino acid will have a
    substution or mutate to a stop under neutral evolution?

    Args:
        parent_codon_idxs (torch.Tensor): The parent codons for each sequence. Shape: (codon_count, 3)
        aa_probs (torch.Tensor): The probability that each site will change to each amino acid. Shape: (codon_count, 20)
    """
    parent_aa_idxs = aa_idxs_of_codon_idxs(parent_codon_idxs)
    p_staying_same = aa_probs[(torch.arange(len(parent_aa_idxs)), parent_aa_idxs)]

    return 1.0 - p_staying_same


def non_stop_neutral_aa_mut_probs(
    nt_parent_idxs: Tensor,
    nt_rates: Tensor,
    nt_csps: Tensor,
    branch_length: float,
    multihit_model=None,
) -> Tensor:
    """For every site, what is the probability that the amino acid will have a non-stop
    substution under neutral evolution?"""
    mut_probs = 1.0 - torch.exp(-branch_length * nt_rates)
    parent_codon_idxs = nt_parent_idxs.reshape(-1, 3)
    aa_probs = neutral_aa_probs(
        parent_codon_idxs,
        mut_probs.reshape(-1, 3),
        nt_csps.reshape(-1, 3, 4),
        multihit_model=multihit_model,
    )

    parent_aa_idxs = aa_idxs_of_codon_idxs(parent_codon_idxs)
    aa_probs[torch.arange(len(parent_aa_idxs)), parent_aa_idxs] = 0.0
    # We exclude stops by summing the probabilities of non-wt aa's, instead of
    # taking 1 - p(wt).
    sums = aa_probs.sum(dim=-1)
    return clamp_probability(sums)


def neutral_aa_mut_probs(
    parent_codon_idxs: Tensor,
    codon_mut_probs: Tensor,
    codon_csps: Tensor,
    multihit_model=None,
) -> Tensor:
    """For every site, what is the probability that the amino acid will have a
    substution or mutate to a stop under neutral evolution?

    This code computes all the probabilities and then indexes into that tensor
    to get the relevant probabilities. This isn't the most efficient way to do
    this, but it's the cleanest. We could make it faster as needed.

    Args:
        parent_codon_idxs (torch.Tensor): The parent codons for each sequence. Shape: (codon_count, 3)
        codon_mut_probs (torch.Tensor): The mutation probabilities for each site in each codon. Shape: (codon_count, 3)
        codon_csps (torch.Tensor): The conditional substitution probabilities for each site in each codon. Shape: (codon_count, 3, 4)

    Returns:
        torch.Tensor: The probability that each site will change to some other amino acid.
                      Shape: (codon_count,)
    """

    aa_probs = neutral_aa_probs(
        parent_codon_idxs,
        codon_mut_probs,
        codon_csps,
        multihit_model=multihit_model,
    )
    mut_probs = mut_probs_of_aa_probs(parent_codon_idxs, aa_probs)
    return mut_probs


def mutsel_log_pcp_probability_of(
    sel_matrix, parent, child, nt_rates, nt_csps, multihit_model=None
):
    """Constructs the log_pcp_probability function specific to given nt_rates and
    nt_csps.

    This function takes log_branch_length as input and returns the log probability of
    the child sequence. It uses log of branch length to ensure non-negativity.
    """

    assert len(parent) % 3 == 0
    assert sel_matrix.shape == (len(parent) // 3, 20)

    parent_idxs = sequences.nt_idx_tensor_of_str(parent)
    child_idxs = sequences.nt_idx_tensor_of_str(child)

    def log_pcp_probability(log_branch_length: torch.Tensor):
        branch_length = torch.exp(log_branch_length)
        nt_mut_probs = 1.0 - torch.exp(-branch_length * nt_rates)

        codon_mutsel, sums_too_big = build_codon_mutsel(
            parent_idxs.reshape(-1, 3),
            nt_mut_probs.reshape(-1, 3),
            nt_csps.reshape(-1, 3, 4),
            sel_matrix,
            multihit_model=multihit_model,
        )

        reshaped_child_idxs = child_idxs.reshape(-1, 3)
        child_prob_vector = codon_mutsel[
            torch.arange(len(reshaped_child_idxs)),
            reshaped_child_idxs[:, 0],
            reshaped_child_idxs[:, 1],
            reshaped_child_idxs[:, 2],
        ]

        child_prob_vector = torch.clamp(child_prob_vector, min=1e-10)

        result = torch.sum(torch.log(child_prob_vector))

        assert torch.isfinite(result)

        return result

    return log_pcp_probability


def find_bracket(
    func,
    start,
    suggested_middle,
    end,
    epsilon=1e-10,
    brute_log_step=0.1,
):
    """Find bracket for optimization. Requirements are start < middle < end and
    func(start) > func(middle) < func(end)

    Attempts to use scipy.optimize.bracket, then falls back to brute force search if necessary.
    Args:
        func: The function to optimize.
        start: The minimum value expected for the minimizer.
        suggested_middle: A suggested starting point for the middle value.
        end: The maximum value expected for the minimizer.
        epsilon: A small value to avoid false positives finding minimum function values in brute force search.
        brute_log_step: Step size for brute force search.

    Returns:
        A tuple containing the start, middle, and end points of the bracket.
        start and end may be outside the range of the passed start and end, but middle point will be contained in the passed range.
    """
    if suggested_middle <= start:
        suggested_middle = start + 1 + brute_log_step
    if suggested_middle >= end:
        end = suggested_middle + 2

    try_vals = [
        [suggested_middle - 0.01, suggested_middle + 0.01],
        [suggested_middle - 0.1, suggested_middle + 0.1],
        [
            np.log(max(epsilon, np.exp(suggested_middle) - 0.01)),
            np.log(np.exp(suggested_middle) + 0.01),
        ],
        [suggested_middle - 0.5, suggested_middle + 0.5],
        [suggested_middle - 1.0, suggested_middle],
        [suggested_middle + 1.0, suggested_middle],
        [suggested_middle - 2.0, suggested_middle],
        [suggested_middle + 2.0, suggested_middle],
        [(end - start) / 2, suggested_middle],
        [-3.0, -0.5],
        [start, suggested_middle],
        [suggested_middle, end],
    ]
    for vals in try_vals:
        vals.sort()
        try:
            bracket = optimize.bracket(
                func,
                xa=vals[0],
                xb=vals[1],
            )
        except RuntimeError:
            warn("Bracket failed. Trying again.")
            continue
        # Unfortunately the order of the bracket is not guaranteed:
        nstart, nmid, nend = sorted(bracket[:3])
        if start < nmid and nmid < end:
            return nstart, nmid, nend

    # Now try to find bracket by brute force:
    # 1. Start with the suggested middle point
    # 2. Move left and right until we find a valid bracket
    # 3. If we can't find a valid bracket before hitting both start and end, raise an error

    epsilon = 1e-10
    best_x = suggested_middle
    best_y = func(suggested_middle)
    left_x = suggested_middle
    right_x = suggested_middle
    for _ in range(4000):
        if left_x > start:
            left_x = left_x - brute_log_step
            left_y = func(left_x)
            if left_y < best_y:
                best_x = left_x
                best_y = left_y
        if right_x < end:
            right_x = right_x + brute_log_step
            right_y = func(right_x)
            if right_y < best_y:
                best_x = right_x
                best_y = right_y
        if (
            not np.isclose(best_x, left_x)
            and not np.isclose(best_x, right_x)
            and best_y + epsilon < left_y
            and best_y + epsilon < right_y
        ):
            return (left_x, best_x, right_x)
        elif left_x < start and right_x > end:
            break
    raise ValueError(
        "Could not find a valid bracket. " "Try using a different starting point."
    )


def optimize_branch_length(
    log_prob_fn,
    starting_branch_length,
    max_optimization_steps=1000,
    optimization_tol=1e-10,
    log_branch_length_lower_threshold=-14.0,
    **kwargs,
):
    log_starting_branch_length = np.log(starting_branch_length)
    upper_bound = max(0.0, log_starting_branch_length + 1.0)

    def loss_func(x):
        return -log_prob_fn(torch.tensor(x)).item()

    try:
        bracket = find_bracket(
            loss_func,
            log_branch_length_lower_threshold,
            log_starting_branch_length,
            upper_bound,
        )
        result = optimize.minimize_scalar(
            loss_func,
            bracket=bracket,
            options={"xtol": optimization_tol, "maxiter": max_optimization_steps},
            method="brent",
        )
    except ValueError:
        raise_upper_bound = True
        while raise_upper_bound:
            result = optimize.minimize_scalar(
                loss_func,
                bounds=(log_branch_length_lower_threshold, upper_bound),
                options={"xatol": optimization_tol, "maxiter": max_optimization_steps},
                method="bounded",
            )
            raise_upper_bound = result.x > upper_bound - 0.1
            upper_bound = upper_bound + 1

    failed_to_converge = not result.success
    if result.x < log_branch_length_lower_threshold + 0.5:
        warn(
            "Optimization result is near lower threshold. This may indicate a problem."
        )
    return np.exp(result.x), failed_to_converge


def set_parent_codon_prob(codon_probs, parent_codon_idxs):
    """Adjust the parent codon probability so that codon probs sum to one at each site.

    Args:
        codon_probs: The codon probabilities in linear space.
            Shape: [B, L, 64]
        parent_codon_idxs (torch.Tensor): The indices of the parent codons.
            Shape: [B, L]
    Returns:
        torch.Tensor: The adjusted codon probabilities.
            Shape: [B, L, 64]
    """
    valid_mask = parent_codon_idxs != sequences.AMBIGUOUS_CODON_IDX  # Shape: [B, L]

    # Zero out the parent indices in codon_probs, while keeping the computation
    # graph intact.
    preds_zeroer = torch.ones_like(codon_probs)
    preds_zeroer[valid_mask, parent_codon_idxs[valid_mask]] = 0.0
    codon_probs = codon_probs * preds_zeroer

    # Calculate the non-parent sum after zeroing out the parent indices.
    non_parent_sum = codon_probs[valid_mask, :].sum(dim=-1)

    # Add these parent values back in, again keeping the computation graph intact.
    preds_parent = torch.zeros_like(codon_probs)
    preds_parent[valid_mask, parent_codon_idxs[valid_mask]] = 1.0 - non_parent_sum
    preds = codon_probs + preds_parent
    # Set ambiguous codons to nan to make sure that we handle them correctly downstream.
    preds[~valid_mask, :] = float("nan")
    return preds


def lift_to_per_aa_selection_factors(selection_factors, aa_parent_idxs):
    """Build a selection matrix from a selection factor tensor for a single sequence.

    upgrades the provided tensor containing a selection factor per site to a matrix
    containing a selection factor per site and amino acid. The wildtype aa selection
    factor is set to 1, and the rest are set to the selection factor.
    """
    selection_matrix = torch.zeros((len(selection_factors), 20), dtype=torch.float)
    # Every "off-diagonal" entry of the selection matrix is set to the selection
    # factor, where "diagonal" means keeping the same amino acid.
    selection_matrix[:, :] = selection_factors[:, None]
    valid_mask = aa_parent_idxs < 20
    selection_matrix[
        torch.arange(len(aa_parent_idxs))[valid_mask], aa_parent_idxs[valid_mask]
    ] = 1.0
    selection_matrix[~valid_mask] = 1.0
    return selection_matrix
