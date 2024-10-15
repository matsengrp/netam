import torch
import numpy as np

from netam.common import BASES


# Define the number of bases (e.g., 4 for DNA/RNA)
_num_bases = 4

# Generate all possible codons using broadcasting
_i, _j, _k = np.indices((_num_bases, _num_bases, _num_bases))  # Create index grids
_codon1 = np.stack((_i, _j, _k), axis=-1)  # Shape: (4, 4, 4, 3)

# Expand dimensions to compare all codon pairs using broadcasting
_codon1_expanded = _codon1[
    :, :, :, np.newaxis, np.newaxis, np.newaxis, :
]  # Shape: (4, 4, 4, 1, 1, 1, 3)
_codon2_expanded = _codon1[
    np.newaxis, np.newaxis, np.newaxis, :, :, :, :
]  # Shape: (1, 1, 1, 4, 4, 4, 3)

# Count the number of differing positions between each pair of codons
"""hit_class_tensor is a tensor of shape (4, 4, 4, 4, 4, 4) recording the hit class (number of nucleotide differences) between all possible parent and child codons. The first three dimensions represent the parent codon, and the last three represent the child codon. Codons are identified by triples of nucleotide indices from `netam.common.BASES`."""
hit_class_tensor = torch.tensor(
    np.sum(_codon1_expanded != _codon2_expanded, axis=-1)
).int()


def parent_specific_hit_classes(parent_codon_idxs: torch.Tensor) -> torch.Tensor:
    """Produce a tensor containing the hit classes of all possible child codons, for
    each passed parent codon.

    Args:
        parent_codon_idxs (torch.Tensor): A (codon_count, 3) shaped tensor containing for each codon, the
            indices of the parent codon's nucleotides.
    Returns:
        torch.Tensor: A (codon_count, 4, 4, 4) shaped tensor containing the hit classes of each possible child codon for each parent codon.
    """
    return hit_class_tensor[
        parent_codon_idxs[:, 0], parent_codon_idxs[:, 1], parent_codon_idxs[:, 2]
    ]


def apply_multihit_correction(
    parent_codon_idxs: torch.Tensor,
    codon_logprobs: torch.Tensor,
    hit_class_factors: torch.Tensor,
) -> torch.Tensor:
    """Multiply codon probabilities by their hit class factors, and renormalize.

    Suppose there are N codons, then the parameters are as follows:

    Args:
        parent_codon_idxs (torch.Tensor): A (N, 3) shaped tensor containing for each codon, the
            indices of the parent codon's nucleotides.
        codon_logprobs (torch.Tensor): A (N, 4, 4, 4) shaped tensor containing the log probabilities
            of mutating to each possible target codon, for each of the N parent codons.
        hit_class_factors (torch.Tensor): A tensor containing the log hit class factors for hit classes 1, 2, and 3. The
            factor for hit class 0 is assumed to be 1 (that is, 0 in log-space).

    Returns:
        torch.Tensor: A (N, 4, 4, 4) shaped tensor containing the log probabilities of mutating to each possible
            target codon, for each of the N parent codons, after applying the hit class factors.
    """
    warnings.warn("hit_class.py:apply_multihit_correction is deprecated, use apply_multihit_correction_nonlog instead")
    per_parent_hit_class = parent_specific_hit_classes(parent_codon_idxs)
    corrections = torch.cat([torch.tensor([0.0]), hit_class_factors])
    reshaped_corrections = corrections[per_parent_hit_class]
    unnormalized_corrected_logprobs = codon_logprobs + reshaped_corrections
    normalizations = torch.logsumexp(
        unnormalized_corrected_logprobs, dim=[1, 2, 3], keepdim=True
    )
    result = unnormalized_corrected_logprobs - normalizations
    if torch.any(torch.isnan(result)):
        print("NAN found in multihit correction application")
        assert False
    return result

def apply_multihit_correction_nonlog(
    parent_codon_idxs: torch.Tensor,
    codon_probs: torch.Tensor,
    hit_class_factors: torch.Tensor,
) -> torch.Tensor:
    """Multiply codon probabilities by their hit class factors, and renormalize.

    Suppose there are N codons, then the parameters are as follows:

    Args:
        parent_codon_idxs (torch.Tensor): A (N, 3) shaped tensor containing for each codon, the
            indices of the parent codon's nucleotides.
        codon_probs (torch.Tensor): A (N, 4, 4, 4) shaped tensor containing the probabilities
            of mutating to each possible target codon, for each of the N parent codons.
        hit_class_factors (torch.Tensor): A tensor containing the log hit class factors for hit classes 1, 2, and 3. The
            factor for hit class 0 is assumed to be 1 (that is, 0 in log-space).

    Returns:
        torch.Tensor: A (N, 4, 4, 4) shaped tensor containing the probabilities of mutating to each possible
            target codon, for each of the N parent codons, after applying the hit class factors.
    """
    per_parent_hit_class = parent_specific_hit_classes(parent_codon_idxs)
    corrections = torch.cat([torch.tensor([0.0]), hit_class_factors]).exp()
    reshaped_corrections = corrections[per_parent_hit_class]
    unnormalized_corrected_probs = codon_probs * reshaped_corrections
    normalizations = torch.sum(
        unnormalized_corrected_probs, dim=[1, 2, 3], keepdim=True
    )
    result = unnormalized_corrected_probs / normalizations
    if torch.any(torch.isnan(result)):
        print("NAN found in multihit correction application")
        assert False
    return result


class MultihitApplier:
    def __init__(self, hit_class_factors, device=None):
        if np.allclose(hit_class_factors, 0):
            warnings.warn("Hit class factors are all zero, and will not change probabilities")
        self.corrections = hit_class_factors.to(device)

    def to(self, device):
        self.corrections = self.corrections.to(device)
        return self

    def __call__(self, parent_codon_idxs, codon_probs):
        return apply_multihit_correction_nonlog(parent_codon_idxs, codon_probs, self.corrections)

    # def __call__(self, parent_codon_idxs, codon_logprobs):
    #     return codon_logprobs
    #     return apply_multihit_correction(parent_codon_idxs, codon_logprobs, self.corrections)


def hit_class_probs_tensor(
    parent_codon_idxs: torch.Tensor, codon_probs: torch.Tensor
) -> torch.Tensor:
    """Calculate probabilities of hit classes between parent codons and all other codons
    for all the sites of a sequence.

    Args:
        parent_codon_idxs (torch.Tensor): The parent nucleotide sequence encoded as a tensor of shape (codon_count, 3),
            containing the nt indices of each codon.
        codon_probs (torch.Tensor): A (codon_count, 4, 4, 4) shaped tensor containing the probabilities of various
            codons, for each codon in parent seq.

    Returns:
        probs (torch.Tensor): A tensor containing the probabilities of different
            counts of hit classes between parent codons and
            all other codons, with shape (codon_count, 4).

    Notes:
    Uses hit_class_tensor (torch.Tensor): A 4x4x4x4x4x4 tensor which when indexed with a parent codon produces
    the hit classes to all possible child codons.
    """

    # Get a codon_countx4x4x4 tensor describing for each parent codon the hit classes of all child codons
    per_parent_hit_class = parent_specific_hit_classes(parent_codon_idxs)
    codon_count = per_parent_hit_class.size(0)
    hc_prob_tensor = torch.zeros(codon_count, 4)
    for k in range(4):
        mask = per_parent_hit_class == k
        hc_prob_tensor[:, k] = (codon_probs * mask).sum(dim=(1, 2, 3))

    return hc_prob_tensor
