from framework import Burrito, trimmed_shm_model_outputs_of_crepe
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

from epam.molevol import (
    reshape_for_codons,
    build_mutation_matrices,
    codon_probs_of_mutation_matrices,
)
from epam.torch_common import optimize_branch_length
from epam import sequences
from netam.common import nt_mask_tensor_of, BASES, stack_heterogeneous
from netam.framework import encode_mut_pos_and_base

def hit_class(codon1, codon2):
    return sum(c1 != c2 for c1, c2 in zip(codon1, codon2))

hit_class_tensors = {}

# Iterate over all possible codons and calculate the hit_class_tensors
for i, base1 in enumerate(BASES):
    for j, base2 in enumerate(BASES):
        for k, base3 in enumerate(BASES):
            codon = base1 + base2 + base3
            hit_class_tensor = torch.zeros(4, 4, 4, dtype=torch.int)
            for i2, base1_2 in enumerate(BASES):
                for j2, base2_2 in enumerate(BASES):
                    for k2, base3_2 in enumerate(BASES):
                        codon_2 = base1_2 + base2_2 + base3_2
                        hit_class_tensor[i2, j2, k2] = hit_class(codon, codon_2)
            hit_class_tensors[codon] = hit_class_tensor

# make a dict mapping from codon to triple integer index
codon_to_idxs = {base_1+base_2+base_3: (i, j, k) for i, base_1 in enumerate(BASES) for j, base_2 in enumerate(BASES) for k, base_3 in enumerate(BASES)}

def hit_class_probs(hit_class_tensor, codon_probs):
    """
    Calculate total probabilities for each number of differences between codons.

    Args:
    - hit_class_tensor (torch.Tensor): A 4x4x4 integer tensor containing the number of differences
                                       between each codon and a reference codon.
    - codon_probs (torch.Tensor): A 4x4x4 tensor containing the probabilities of various codons.

    Returns:
    - total_probs (torch.Tensor): A 1D tensor containing the total probabilities for each number
                                   of differences (0 to 3).
    """
    total_probs = []

    for hit_class in range(4):
        # Create a mask of codons with the desired number of differences
        mask = hit_class_tensor == hit_class

        # Multiply componentwise with the codon_probs tensor and sum
        total_prob = (codon_probs * mask.float()).sum()

        # Append the total probability to the list
        total_probs.append(total_prob.item())

    return torch.tensor(total_probs)


def hit_class_probs_seq(parent_seq, codon_probs):
    """
    Calculate probabilities of hit classes between parent codons and all other codons for all the sites of a sequence.

    Args:
    - parent_seq (str): The parent nucleotide sequence.
    - codon_probs (torch.Tensor): A tensor containing the probabilities of various codons.

    Returns:
    - probs (torch.Tensor): A tensor containing the probabilities of different
                            counts of hit classes between parent codons and
                            all other codons.
    
    Notes:

    Uses hit_class_tensors (dict): A dictionary containing hit_class_tensors indexed by codons.
    """
    # Check if the size of the first dimension of codon_probs matches the length of parent_seq divided by 3
    if len(parent_seq) // 3 != codon_probs.size(0):
        raise ValueError(
            "The size of the first dimension of codon_probs should match the length of parent_seq divided by 3."
        )

    # Initialize a list to store the probabilities of different counts of differences
    probs = []

    # Iterate through codons in parent_seq
    for i in range(0, len(parent_seq), 3):
        # Extract the codon from parent_seq
        codon = parent_seq[i : i + 3]

        # if codon contains an N, append a tensor of 4 -1s to probs then continue
        if "N" in codon:
            probs.append(torch.tensor([-100.0] * 4))
            continue

        # Get the corresponding hit_class_tensor from hit_class_tensors
        hit_class_tensor = hit_class_tensors[codon]

        # Get the ith entry of codon_probs
        codon_probs_i = codon_probs[i // 3]

        # Calculate the probabilities of different counts of differences using the hit_class_tensor and codon_probs_i
        total_probs = hit_class_probs(hit_class_tensor, codon_probs_i)

        # Append the probabilities to the list
        probs.append(total_probs)

    # Concatenate all the probabilities into a tensor
    probs = torch.stack(probs)

    return probs

def codon_probs_of_parent_scaled_rates_and_sub_probs(
    parent_idxs, scaled_rates, sub_probs
):
    """
    Compute the probabilities of mutating to various codons for a parent sequence.

    This uses the same machinery as we use for fitting the DNSM, but we stay on
    the codon level rather than moving to syn/nonsyn changes.
    """
    # This is from `aaprobs_of_parent_scaled_rates_and_sub_probs`:
    mut_probs = 1.0 - torch.exp(-scaled_rates)
    parent_codon_idxs = reshape_for_codons(parent_idxs)
    codon_mut_probs = reshape_for_codons(mut_probs)
    codon_sub_probs = reshape_for_codons(sub_probs)

    # This is from `aaprob_of_mut_and_sub`:
    mut_matrices = build_mutation_matrices(
        parent_codon_idxs, codon_mut_probs, codon_sub_probs
    )
    codon_probs = codon_probs_of_mutation_matrices(mut_matrices)

    return codon_probs


def _trim_seqs_to_codon_boundary_and_max_len(seqs, site_count):
    """Assumes that all sequences have the same length"""
    max_len = site_count - site_count % 3
    return [seq[: min(len(seq) - len(seq) % 3, max_len)] for seq in seqs]


def _prepare_pcp_df(pcp_df, crepe, site_count):
    """
    Trim the sequences to codon boundaries and add the rates and substitution probabilities.
    """
    pcp_df["parent"] = _trim_seqs_to_codon_boundary_and_max_len(
        pcp_df["parent"], site_count
    )
    pcp_df["child"] = _trim_seqs_to_codon_boundary_and_max_len(
        pcp_df["child"], site_count
    )
    pcp_df = pcp_df[pcp_df["parent"] != pcp_df["child"]].reset_index(drop=True)
    ratess, cspss = trimmed_shm_model_outputs_of_crepe(crepe, pcp_df["parent"])
    pcp_df["rates"] = ratess
    pcp_df["subs_probs"] = cspss
    return pcp_df

class CodonProbDataset(Dataset):
    def __init__(
        self,
        nt_parents,
        nt_children,
        all_rates,
        all_subs_probs,
        branch_length_multiplier=5.0,
    ):
        self.nt_parents = nt_parents
        self.nt_children = nt_children
        self.all_rates = stack_heterogeneous(all_rates.reset_index(drop=True))
        self.all_subs_probs = stack_heterogeneous(all_subs_probs.reset_index(drop=True))

        assert len(self.nt_parents) == len(self.nt_children)
        pcp_count = len(self.nt_parents)

        for parent, child in zip(self.nt_parents, self.nt_children):
            if parent == child:
                raise ValueError(
                    f"Found an identical parent and child sequence: {parent}"
                )

        # Insert here hit classes and codon probs, as attributes...
        # see neutral codon notebook use of hc_site_df_of_pcp_df

        # Make initial branch lengths (will get optimized later).
        self._branch_lengths = np.array(
            [
                sequences.nt_mutation_frequency(parent, child)
                * branch_length_multiplier
                for parent, child in zip(self.nt_parents, self.nt_children)
            ]
        )
        self.update_hit_classes()

    @property
    def branch_lengths(self):
        return self._branch_lengths

    @branch_lengths.setter
    def branch_lengths(self, new_branch_lengths):
        assert len(new_branch_lengths) == len(self._branch_lengths), (
            f"Expected {len(self._branch_lengths)} branch lengths, "
            f"got {len(new_branch_lengths)}"
        )
        assert np.all(np.isfinite(new_branch_lengths) & (new_branch_lengths > 0))
        self._branch_lengths = new_branch_lengths
        self.update_hit_classes()
    
    def update_hit_classes(self):
        """Compute hit class probabilities for all codons in each sequence based on current branch lengths"""
        raise NotImplementedError
        # see neutral codon notebook use of hc_site_df_of_pcp_df

class CodonProbBurrito(Burrito):
    def __init__(
        self,
        train_dataset: CodonProbDataset,
        val_dataset: CodonProbDataset,
        model,
        **kwargs,
    ):
        super().__init__(
            train_dataset,
            val_dataset,
            model,
            **kwargs,
        )

    # These are from DNSMBurrito, as a start
    def _find_optimal_branch_length(
        self,
        parent,
        child,
        rates,
        subs_probs,
        starting_branch_length,
        **optimization_kwargs,
    ):

        # truncate each to be a multiple of 3 in length
        # (or, this may have already been done TODO)
        # From hit_class_probs_of_pcp_df

        parent = parent[: len(parent) - len(parent) % 3]
        rates = rates[: len(rates) - len(rates) % 3]
        subs_probs = subs_probs[: len(subs_probs) - len(subs_probs) % 3]

        parent_idxs = sequences.nt_idx_tensor_of_str(parent.replace("N", "A"))
        # From issue: "Just to get things going, let's have our log probability just be the probability of getting a codon mutation vs no."
        def log_pcp_probability(log_branch_length):
            branch_length = torch.exp(log_branch_length)
            scaled_rates = branch_length * rates

            codon_probs = codon_probs_of_parent_scaled_rates_and_sub_probs(
                parent_idxs, scaled_rates, subs_probs
            )

            hc_probs = hit_class_probs_seq(parent, codon_probs, hit_class_tensors)
            # ^^ contains for all parent codons the probabilities of different counts of hit classes to all other codons
            # We want probability of getting at least one codon mutation across the whole sequence?
            # That would be 1 - prod([it[0] for it in hc_probs])
            return

        return optimize_branch_length(
            log_pcp_probability,
            starting_branch_length.double().item(),
            **optimization_kwargs,
        )

    def find_optimal_branch_lengths(self, dataset, **optimization_kwargs):
        optimal_lengths = []

        for parent, child, rates, subs_probs, starting_length in tqdm(
            zip(
                dataset.nt_parents,
                dataset.nt_children,
                dataset.all_rates,
                dataset.all_subs_probs,
                dataset.branch_lengths,
            ),
            total=len(dataset.nt_parents),
            desc="Finding optimal branch lengths",
        ):
            optimal_lengths.append(
                self._find_optimal_branch_length(
                    parent,
                    child,
                    rates[: len(parent)],
                    subs_probs[: len(parent), :],
                    starting_length,
                    **optimization_kwargs,
                )
            )

        return np.array(optimal_lengths)


# class CodonProbDataset(Dataset):
#     def __init__(self, pcp_df, crepe, site_count):
#         super().__init__()
#         pcp_df = _prepare_pcp_df(pcp_df, crepe, site_count)
#         assert pcp_df["parent"].apply(len).max() <= site_count - site_count % 3

#         # This constructor will take a data frame containing the following, and split it into series which will be stored as attributes on each instance.
#         # First they'll be processed a bit, and we'll also compute the hit class probs seq for each pcp. Then we can remove some of that code from the branch length optimization
#         # method above.
#         parent, rates, subs_probs, branch_length = pcp_df.loc[0, ["parent", "rates", "subs_probs", "branch_length"]]
#         # truncate each to be a multiple of 3 in length
#         parent = parent[:len(parent) - len(parent) % 3]
#         rates = rates[:len(rates) - len(rates) % 3]
#         subs_probs = subs_probs[:len(subs_probs) - len(subs_probs) % 3]

#         mask = nt_mask_tensor_of(parent)
#         parent_idxs = sequences.nt_idx_tensor_of_str(parent.replace("N", "A"))
#         parent_len = len(parent)
#         scaled_rates = branch_length * rates[:parent_len]

#         codon_probs = codon_probs_of_parent_scaled_rates_and_sub_probs(parent_idxs, scaled_rates, subs_probs)


