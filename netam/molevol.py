"""Free functions for molecular evolution computation.

CSP means conditional substitution probability. CSPs are the probabilities of alternate
states conditioned on there being a substitution.
"""

import numpy as np

import torch
from torch import Tensor, optim

from netam.sequences import CODON_AA_INDICATOR_MATRIX

import netam.sequences as sequences


def check_csps(parent_idxs: Tensor, csps: Tensor) -> Tensor:
    """Make sure that the CSPs are valid, i.e. that they are a probability distribution
    and the parent state is zero.

    Args:
        parent_idxs (torch.Tensor): The parent sequence indices.
        sub_probs (torch.Tensor): A 2D PyTorch tensor representing substitution
            probabilities. Rows correspond to sites, and columns correspond
            to states (e.g. nucleotides).
    """

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
    )


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
        codon_probs = multihit_model(parent_codon_idxs, codon_probs)

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


def neutral_aa_probs(
    parent_codon_idxs: Tensor,
    codon_mut_probs: Tensor,
    codon_csps: Tensor,
    multihit_model=None,
) -> Tensor:
    """For every site, what is the probability that the amino acid will mutate to every
    amino acid?

    Args:
        parent_codon_idxs (torch.Tensor): The parent codons for each sequence. Shape: (codon_count, 3)
        codon_mut_probs (torch.Tensor): The mutation probabilities for each site in each codon. Shape: (codon_count, 3)
        codon_csps (torch.Tensor): The substitution probabilities for each site in each codon. Shape: (codon_count, 3, 4)

    Returns:
        torch.Tensor: The probability that each site will change to each amino acid.
                      Shape: (codon_count, 20)
    """

    mut_matrices = build_mutation_matrices(
        parent_codon_idxs, codon_mut_probs, codon_csps
    )
    codon_probs = codon_probs_of_mutation_matrices(mut_matrices)

    if multihit_model is not None:
        codon_probs = multihit_model(parent_codon_idxs, codon_probs)

    # Get the probability of mutating to each amino acid.
    aa_probs = codon_probs.view(-1, 64) @ CODON_AA_INDICATOR_MATRIX

    return aa_probs


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
    # We build a table that will allow us to look up the amino acid index
    # from the codon indices. Argmax gets the aa index.
    aa_idx_from_codon = CODON_AA_INDICATOR_MATRIX.argmax(dim=1).view(4, 4, 4)

    # Get the amino acid index for each parent codon.
    parent_aa_idxs = aa_idx_from_codon[
        (
            parent_codon_idxs[:, 0],
            parent_codon_idxs[:, 1],
            parent_codon_idxs[:, 2],
        )
    ]
    p_staying_same = aa_probs[(torch.arange(len(parent_aa_idxs)), parent_aa_idxs)]

    return 1.0 - p_staying_same


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

        # This is a diagnostic generating data for netam issue #7.
        # if sums_too_big is not None:
        #     self.csv_file.write(f"{parent},{child},{branch_length},{sums_too_big}\n")

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


def optimize_branch_length(
    log_prob_fn,
    starting_branch_length,
    learning_rate=0.1,
    max_optimization_steps=1000,
    optimization_tol=1e-3,
    log_branch_length_lower_threshold=-10.0,
):
    log_branch_length = torch.tensor(np.log(starting_branch_length), requires_grad=True)

    optimizer = optim.Adam([log_branch_length], lr=learning_rate)
    prev_log_branch_length = log_branch_length.clone()

    step_idx = 0

    for step_idx in range(max_optimization_steps):
        # For some PCPs, the optimizer works very hard optimizing very tiny branch lengths.
        if log_branch_length < log_branch_length_lower_threshold:
            break

        optimizer.zero_grad()

        loss = -log_prob_fn(log_branch_length)
        assert torch.isfinite(
            loss
        ), f"Loss is not finite on step {step_idx}: perhaps selection has given a probability of zero?"
        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_branch_length], max_norm=5.0)
        optimizer.step()
        if torch.isnan(log_branch_length):
            raise ValueError("branch length optimization resulted in NAN")

        change_in_log_branch_length = torch.abs(
            log_branch_length - prev_log_branch_length
        )
        if change_in_log_branch_length < optimization_tol:
            break

        prev_log_branch_length = log_branch_length.clone()

    if step_idx == max_optimization_steps - 1:
        print(
            f"Warning: optimization did not converge after {max_optimization_steps} steps; log branch length is {log_branch_length.detach().item()}"
        )
        failed_to_converge = True
    else:
        failed_to_converge = False

    return torch.exp(log_branch_length.detach()).item(), failed_to_converge
