import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd

from netam.molevol import (
    reshape_for_codons,
    build_mutation_matrices,
    codon_probs_of_mutation_matrices,
    optimize_branch_length,
)
from netam import sequences
from netam.common import BASES, stack_heterogeneous, clamp_probability
import netam.framework as framework
from netam.framework import Burrito
from netam.models import AbstractBinarySelectionModel

def hit_class(codon1, codon2):
    return sum(c1 != c2 for c1, c2 in zip(codon1, codon2))


# Initialize the 4D tensor to store the hit class tensors
# The shape is [4, 4, 4, 4, 4, 4], corresponding to three nucleotide indices and the hit class tensor (4x4x4)
num_bases = len(BASES)
hit_class_tensor_full = torch.zeros(num_bases, num_bases, num_bases, num_bases, num_bases, num_bases, dtype=torch.int)

# Populate the tensor
for i,_ in enumerate(BASES):
    for j, _ in enumerate(BASES):
        for k, _ in enumerate(BASES):
            codon1 = (i, j, k)
            for i2, _ in enumerate(BASES):
                for j2, _ in enumerate(BASES):
                    for k2, _ in enumerate(BASES):
                        codon_2 = (i2, j2, k2)
                        hit_class_tensor_full[i, j, k, i2, j2, k2] = hit_class(codon1, codon_2)


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

def hit_class_probs_tensor(parent_codon_idxs, codon_probs):
    """
    Calculate probabilities of hit classes between parent codons and all other codons for all the sites of a sequence.

    Args:
    - parent_codon_idxs (torch.Tensor): The parent nucleotide sequence encoded as a tensor of length Cx3, containing the nt indices of each codon.
    - codon_probs (torch.Tensor): A Cx4x4x4 tensor containing the probabilities of various codons, for each codon in parent seq.

    Returns:
    - probs (torch.Tensor): A tensor containing the probabilities of different
                            counts of hit classes between parent codons and
                            all other codons.
    
    Notes:

    Uses hit_class_tensor_full (torch.Tensor): A 4x4x4x4x4x4 tensor which when indexed with a parent codon produces
    the hit classes to all possible child codons.
    """

    # Get a Cx4x4x4 tensor describing for each parent codon the hit classes of all child codons
    hit_class_tensor_t = hit_class_tensor_full[parent_codon_idxs[:, 0], 
                                               parent_codon_idxs[:, 1], 
                                               parent_codon_idxs[:, 2]].int()
    C = hit_class_tensor_t.size(0)
    hc_prob_tensor = torch.zeros(C, 4)
    for k in range(4):
        mask = (hit_class_tensor_t == k)
        hc_prob_tensor[:, k] = (codon_probs * mask).sum(dim=(1,2,3))

    return hc_prob_tensor


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

# From hit_classes_of_pcp_df
def _observed_hit_classes(parents, children):
    labels = []
    for parent_seq, child_seq in zip(parents, children):

        assert len(parent_seq) == len(child_seq)
        codon_count = len(parent_seq) // 3
        valid_length =  codon_count * 3

        # Chunk into codons and count mutations
        num_mutations = []
        for i in range(0, valid_length, 3):
            parent_codon = parent_seq[i : i + 3]
            child_codon = child_seq[i : i + 3]

            if "N" in parent_codon or "N" in child_codon:
                num_mutations.append(-100)
            else:
                # Count differing bases
                mutations = sum(1 for p, c in zip(parent_codon, child_codon) if p != c)
                num_mutations.append(mutations)

        # Pad or truncate the mutation counts to match codon_count
        padded_mutations = num_mutations[:codon_count]  # Truncate if necessary
        padded_mutations += [-100] * (
            codon_count - len(padded_mutations)
        )  # Pad with -1s

        # Update the labels tensor for this row
        labels.append(torch.tensor(padded_mutations, dtype=torch.int))
    return labels



class CodonProbDataset(Dataset):
    def __init__(
        self,
        nt_parents,
        nt_children,
        all_rates,
        all_subs_probs,
        branch_length_multiplier=1.0,
    ):
        #TODO figure out if we need a branch_length_multiplier, or if normalized mutation frequency is fine
        #TODO mask or whatever to account for this replacement of N's with A's.
        trimmed_parents = [parent[: len(parent) - len(parent) % 3] for parent in nt_parents]
        trimmed_children = [child[: len(child) - len(child) % 3] for child in nt_children]
        self.nt_parents = stack_heterogeneous(pd.Series(sequences.nt_idx_tensor_of_str(parent.replace("N", "A")) for parent in trimmed_parents))
        self.nt_children = stack_heterogeneous(pd.Series(sequences.nt_idx_tensor_of_str(child.replace("N", "A")) for child in trimmed_children))
        max_len = len(self.nt_parents[0])
        self.nt_parents_strs = [parent + ("N" * (max_len - len(parent))) for parent in trimmed_parents]
        self.nt_children_strs = [child + ("N" * (max_len - len(child))) for child in trimmed_children]
        self.all_rates = stack_heterogeneous(pd.Series(rates[: len(rates) - len(rates) % 3] for rates in all_rates).reset_index(drop=True))
        self.all_subs_probs = stack_heterogeneous(pd.Series(subs_probs[: len(subs_probs) - len(subs_probs) % 3] for subs_probs in all_subs_probs).reset_index(drop=True))

        assert len(self.nt_parents) == len(self.nt_children)

        # TODO get hit classes and do checks directly from tensor encoding of sequences
        for parent, child in zip(trimmed_parents, trimmed_children):
            if parent == child:
                raise ValueError(
                    f"Found an identical parent and child sequence: {parent}"
                )
            assert len(parent) == len(child)

        self.observed_hcs = stack_heterogeneous(_observed_hit_classes(trimmed_parents, trimmed_children), pad_value=-100).long()
        # At some point, we may want a nt_parents mask for N's, but this ignores codons with N's, anyway.
        self.codon_mask = self.observed_hcs > -1

        # Make initial branch lengths (will get optimized later).
        self._branch_lengths = torch.tensor(
            [
                sequences.nt_mutation_frequency(parent, child)
                * branch_length_multiplier
                for parent, child in zip(trimmed_parents, trimmed_children)
            ]
        )
        self.update_hit_class_probs()

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
        self.update_hit_class_probs()
    
    def update_hit_class_probs(self):
        """Compute hit class probabilities for all codons in each sequence based on current branch lengths"""
        new_hc_probs = []
        for (
            encoded_parent,
            rates,
            subs_probs,
            branch_length,
        ) in zip(
            self.nt_parents,
            self.all_rates,
            self.all_subs_probs,
            self.branch_lengths,
        ):
            # This encodes bases as indices in a sorted nucleotide list. Codons containing
            # N's should already be masked in self.codon_mask, so treating them as A's here shouldn't matter...
            # TODO Check that assertion ^^

            scaled_rates = branch_length * rates

            codon_probs = codon_probs_of_parent_scaled_rates_and_sub_probs(
                encoded_parent, scaled_rates[:len(encoded_parent)], subs_probs[:len(encoded_parent)]
            )

            new_hc_probs.append(hit_class_probs_tensor(reshape_for_codons(encoded_parent), codon_probs))
        # We must store probability of all hit classes for arguments to cce_loss in loss_of_batch.
        self.hit_class_probs = stack_heterogeneous(new_hc_probs, pad_value=-100)

    # A couple of these methods could be moved to a super class, which itself subclasses Dataset
    def export_branch_lengths(self, out_csv_path):
        pd.DataFrame({"branch_length": self.branch_lengths}).to_csv(
            out_csv_path, index=False
        )

    def load_branch_lengths(self, in_csv_path):
        self.branch_lengths = torch.tensor(pd.read_csv(in_csv_path)["branch_length"].values)

    def __len__(self):
        return len(self.nt_parents)

    def __getitem__(self, idx):
        return {
            "parent": self.nt_parents[idx],
            "child": self.nt_children[idx],
            "observed_hcs": self.observed_hcs[idx],
            "rates": self.all_rates[idx],
            "subs_probs": self.all_subs_probs[idx],
            "hit_class_probs": self.hit_class_probs[idx],
            "codon_mask": self.codon_mask[idx],
        }

    def to(self, device):
        # TODO update this (and might have to encode sequences as Tensors), if used!
        raise NotImplementedError
        self.codon_mask = self.mask.to(device)
        self.all_rates = self.all_rates.to(device)
        self.all_subs_probs = self.all_subs_probs.to(device)
        self.hit_class_probs = self.hit_class_probs.to(device)

def flatten_and_mask_sequence_codons(input_tensor, codon_mask=None):
        """Flatten first dimension, that is over sequences, to return tensor
        whose first dimension contains all codons, instead of sequences.

        input_tensor should have at least two dimensions, with the second dimension representing codons
        codon_mask should have two dimensions
        
        If mask is provided, also mask the result"""
        # If N is number pcps, and C is number codons per seq, then
        # hit_class_probs is of dimension (N, C, 4)
        # codon_mask is of dimension (N, C)
        # observed_hcs is of dimension (N, C)

        # Since there's a different number of codons in each sequence, we need to flatten so we just
        # have a big list of codons, before we do masking (otherwise we'll get everything flattened to a single dimension)
        flat_input = input_tensor.flatten(start_dim=0, end_dim=1)
        if codon_mask is not None:
            flat_codon_mask = codon_mask.flatten()
            flat_input = flat_input[flat_codon_mask]

        # Now this should be shape (N*C, 4) or (N*C,)
        return flat_input

class HitClassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.values = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

    @property
    def hyperparameters(self):
        return {}

    def forward(self, parent_codon_idxs: torch.Tensor, uncorrected_log_codon_probs: torch.Tensor):
        hit_class_tensor_t = hit_class_tensor_full[parent_codon_idxs[:, 0], 
                                                   parent_codon_idxs[:, 1], 
                                                   parent_codon_idxs[:, 2]].int()
        corrections = torch.cat([torch.tensor([0.0]), self.values])
        adjustments = corrections[hit_class_tensor_t]
        unnormalized_corrected_probs = uncorrected_log_codon_probs + adjustments
        normalizations = torch.logsumexp(unnormalized_corrected_probs, dim=[1,2,3], keepdim=True)
        return unnormalized_corrected_probs - normalizations


class CodonProbBurrito(Burrito):
    def __init__(
        self,
        train_dataset: CodonProbDataset,
        val_dataset: CodonProbDataset,
        model: HitClassModel,
        *args,
        **kwargs,
    ):
        super().__init__(
            train_dataset,
            val_dataset,
            model,
            *args,
            **kwargs,
        )
        self.cce_loss = torch.nn.CrossEntropyLoss(reduction='mean')


    # For loss want categorical cross-entropy, appears in framework.py for another model
    # When computing overall log-likelihood will need to account for the different sequence lengths
    def load_branch_lengths(self, in_csv_prefix):
        if self.train_loader is not None:
            self.train_loader.dataset.load_branch_lengths(
                in_csv_prefix + ".train_branch_lengths.csv"
            )
        self.val_loader.dataset.load_branch_lengths(
            in_csv_prefix + ".val_branch_lengths.csv"
        )

    # Once optimized branch lengths, store the baseline codon-level predictions somewhere. See DNSMBurrito::predictions_of_batch
    # Rates stay same, and are used to re-compute branch lengths whenever codon probs are adjusted.
    def loss_of_batch(self, batch):
        # different sequence lengths, and codons containing N's, are marked in the mask.
        observed_hcs = batch["observed_hcs"]
        hit_class_probs = batch["hit_class_probs"]
        codon_mask = batch["codon_mask"]

        flat_masked_hit_class_probs = flatten_and_mask_sequence_codons(hit_class_probs, codon_mask=codon_mask)
        flat_masked_observed_hcs = flatten_and_mask_sequence_codons(observed_hcs, codon_mask=codon_mask).long()
        corrections = torch.cat([torch.tensor([0.0]), self.model.values])
        corrected_probs = flat_masked_hit_class_probs.log() + corrections
        corrected_probs = (corrected_probs - torch.logsumexp(corrected_probs, dim=1, keepdim=True)).exp()
        assert torch.isfinite(corrected_probs).all()
        adjusted_probs = clamp_probability(corrected_probs)

        # Just need to adjust hit class probs by model coefficients, and re-normalize.

        return self.cce_loss(adjusted_probs, flat_masked_observed_hcs)
        # nt_parents = batch["nt_parents"]
        # nt_children = batch["nt_children"]
        # brlens = batch["branch_lengths"]
        # codon_mask = batch["codon_mask"]
        # rates = batch["rates"]
        # subs_probs = batch["subs_probs"]
        # scaled_rates = rates * brlens
        # codon_probs = torch.tensor([codon_probs_of_parent_scaled_rates_and_sub_probs(parent_idxs, scaled_rates_it, subs_probs_it)
        #                             for parent_idxs, scaled_rates_it, subs_probs_it in zip(nt_parents, scaled_rates, subs_probs)])



    # These are from DNSMBurrito, as a start
    def _find_optimal_branch_length(
        self,
        parent_idxs,
        child_idxs,
        observed_hcs,
        rates,
        subs_probs,
        codon_mask,
        starting_branch_length,
        **optimization_kwargs,
    ):

        # # A stand-in for the adjustment model we're fitting:
        # codon_adjustment = self.model.values

        def log_pcp_probability(log_branch_length):
            # We want to first return the log-probability of the observed branch, using codon probs.
            # Then we'll want to adjust codon probs using our hit class probabilities
            branch_length = torch.exp(log_branch_length)
            scaled_rates = rates * branch_length
            # Rates is a 1d tensor containing one rate for each nt site.

            # Codon probs is a Cx4x4x4 tensor containing for each codon idx the
            # distribution on possible target codons (all 64 of them!)
            codon_probs = codon_probs_of_parent_scaled_rates_and_sub_probs(
                parent_idxs, scaled_rates, subs_probs
            )[codon_mask]

            child_codon_idxs = reshape_for_codons(child_idxs)[codon_mask]
            parent_codon_idxs = reshape_for_codons(parent_idxs)[codon_mask]
            corrected_codon_probs = self.model(parent_codon_idxs, codon_probs.log())
            child_codon_probs = corrected_codon_probs[torch.arange(child_codon_idxs.size(0)), child_codon_idxs[:, 0], child_codon_idxs[:, 1], child_codon_idxs[:, 2]]
            return child_codon_probs.sum()

            # # hc_probs is a Cx4 tensor containing codon probs aggregated by hit class
            # hc_probs = hit_class_probs_tensor(parent_codon_idxs, codon_probs)

            # # Add fixed 1 adjustment for hit class 0:
            # _adjust = torch.cat([torch.tensor([1]), codon_adjustment])
            # # Get adjustments for each site's observed hit class
            # observed_hc_adjustments = _adjust.gather(0, observed_hcs[codon_mask])
            # numerators = (child_codon_probs * observed_hc_adjustments).log()
            # # This is a dot product of the distribution and the adjustments at each site
            # denominators = (torch.matmul(hc_probs, _adjust)).log()
            # return (numerators - denominators).sum()


        return optimize_branch_length(
            log_pcp_probability,
            starting_branch_length.double().item(),
            **optimization_kwargs,
        )[0]

    def find_optimal_branch_lengths(self, dataset, **optimization_kwargs):
        optimal_lengths = []

        for parent_idxs, child_idxs, observed_hcs, rates, subs_probs, codon_mask, starting_length in tqdm(
            zip(
                dataset.nt_parents,
                dataset.nt_children,
                dataset.observed_hcs,
                dataset.all_rates,
                dataset.all_subs_probs,
                dataset.codon_mask,
                dataset.branch_lengths,
            ),
            total=len(dataset.nt_parents),
            desc="Optimizing branch lengths",
        ):
            optimal_lengths.append(
                self._find_optimal_branch_length(
                    parent_idxs,
                    child_idxs,
                    observed_hcs,
                    rates[: len(parent_idxs)],
                    subs_probs[: len(parent_idxs), :],
                    codon_mask,
                    starting_length,
                    **optimization_kwargs,
                )
            )

        return torch.tensor(optimal_lengths)

    def to_crepe(self):
        training_hyperparameters = {
            key: self.__dict__[key]
            for key in [
                "learning_rate",
            ]
        }
        encoder = framework.PlaceholderEncoder()
        return framework.Crepe(encoder, self.model, training_hyperparameters)

def codon_prob_dataset_from_pcp_df(pcp_df, branch_length_multiplier=1.0):
    nt_parents = pcp_df["parent"].reset_index(drop=True)
    nt_children = pcp_df["child"].reset_index(drop=True)
    rates = pcp_df["rates"].reset_index(drop=True)
    subs_probs = pcp_df["subs_probs"].reset_index(drop=True)

    return CodonProbDataset(
        nt_parents,
        nt_children,
        rates,
        subs_probs,
        branch_length_multiplier=branch_length_multiplier,
    )

def train_test_datasets_of_pcp_df(pcp_df, train_frac=0.8, branch_length_multiplier=1.0):
    nt_parents = pcp_df["parent"].reset_index(drop=True)
    nt_children = pcp_df["child"].reset_index(drop=True)
    rates = pcp_df["rates"].reset_index(drop=True)
    subs_probs = pcp_df["subs_probs"].reset_index(drop=True)

    train_len = int(train_frac * len(nt_parents))
    train_parents, val_parents = nt_parents[:train_len], nt_parents[train_len:]
    train_children, val_children = nt_children[:train_len], nt_children[train_len:]
    train_rates, val_rates = rates[:train_len], rates[train_len:]
    train_subs_probs, val_subs_probs = (
        subs_probs[:train_len],
        subs_probs[train_len:],
    )
    val_dataset = CodonProbDataset(
        val_parents,
        val_children,
        val_rates,
        val_subs_probs,
        branch_length_multiplier=branch_length_multiplier,
    )
    if train_frac == 0.0:
        return None, val_dataset
    # else:
    train_dataset = CodonProbDataset(
        train_parents,
        train_children,
        train_rates,
        train_subs_probs,
        branch_length_multiplier=branch_length_multiplier,
    )
    return val_dataset, train_dataset
