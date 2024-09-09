import torch
from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd

from netam.molevol import (
    reshape_for_codons,
    build_mutation_matrices,
    codon_probs_of_mutation_matrices,
    optimize_branch_length,
    codon_probs_of_parent_scaled_rates_and_sub_probs,
    hit_class_probs_tensor,
)
from netam import sequences
from netam.common import BASES, stack_heterogeneous, clamp_probability
import netam.framework as framework
from netam.framework import Burrito
from netam.models import HitClassModel


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
        self.all_rates = stack_heterogeneous(pd.Series(rates[: len(rates) - len(rates) % 3] for rates in all_rates).reset_index(drop=True))
        self.all_subs_probs = stack_heterogeneous(pd.Series(subs_probs[: len(subs_probs) - len(subs_probs) % 3] for subs_probs in all_subs_probs).reset_index(drop=True))

        assert len(self.nt_parents) == len(self.nt_children)

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
        assert (torch.isfinite(new_branch_lengths) & (new_branch_lengths > 0)).all()
        self._branch_lengths = new_branch_lengths
        self.update_hit_class_probs()
    
    def update_hit_class_probs(self):
        """Compute hit class probabilities for all codons in each sequence based on current branch lengths"""
        new_codon_probs = []
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
            scaled_rates = branch_length * rates

            codon_probs = codon_probs_of_parent_scaled_rates_and_sub_probs(
                encoded_parent, scaled_rates[:len(encoded_parent)], subs_probs[:len(encoded_parent)]
            )
            new_codon_probs.append(codon_probs)

            new_hc_probs.append(hit_class_probs_tensor(reshape_for_codons(encoded_parent), codon_probs))
        self.codon_probs = torch.stack(new_codon_probs)
        self.hit_class_probs = torch.stack(new_hc_probs)

    # A couple of these methods could maybe be moved to a super class, which itself subclasses Dataset
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
            "codon_probs": self.codon_probs[idx],
            "codon_mask": self.codon_mask[idx],
        }

    def to(self, device):
        self.nt_parents = self.nt_parents.to(device)
        self.nt_children = self.nt_children.to(device)
        self.observed_hcs = self.observed_hcs.to(device)
        self.all_rates = self.all_rates.to(device)
        self.all_subs_probs = self.all_subs_probs.to(device)
        self.hit_class_probs = self.hit_class_probs.to(device)
        self.codon_mask = self.codon_mask.to(device)
        self.branch_lengths = self.branch_lengths.to(device)

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


    def load_branch_lengths(self, in_csv_prefix):
        if self.train_loader is not None:
            self.train_loader.dataset.load_branch_lengths(
                in_csv_prefix + ".train_branch_lengths.csv"
            )
        self.val_loader.dataset.load_branch_lengths(
            in_csv_prefix + ".val_branch_lengths.csv"
        )

    def loss_of_batch(self, batch):
        child_idxs = batch["child"]
        parent_idxs = batch["parent"]
        codon_probs = batch["codon_probs"]
        codon_mask = batch["codon_mask"]

        flat_masked_codon_probs = flatten_and_mask_sequence_codons(codon_probs, codon_mask=codon_mask)
        # remove first dimension of parent_idxs by concatenating all of its elements
        codon_mask_flat = codon_mask.flatten(start_dim=0, end_dim=1)
        parent_codons_flat = reshape_for_codons(parent_idxs.flatten(start_dim=0, end_dim=1))[codon_mask_flat]
        child_codons_flat = reshape_for_codons(child_idxs.flatten(start_dim=0, end_dim=1))[codon_mask_flat]
        corrected_codon_log_probs = self.model(parent_codons_flat, flat_masked_codon_probs.log())
        child_codon_log_probs = corrected_codon_log_probs[torch.arange(child_codons_flat.size(0)),
                                                  child_codons_flat[:, 0],
                                                  child_codons_flat[:, 1],
                                                  child_codons_flat[:, 2]]
        return -child_codon_log_probs.sum()

    def _find_optimal_branch_length(
        self,
        parent_idxs,
        child_idxs,
        rates,
        subs_probs,
        codon_mask,
        starting_branch_length,
        **optimization_kwargs,
    ):

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
            corrected_codon_log_probs = self.model(parent_codon_idxs, codon_probs.log())
            child_codon_log_probs = corrected_codon_log_probs[torch.arange(child_codon_idxs.size(0)),
                                                      child_codon_idxs[:, 0],
                                                      child_codon_idxs[:, 1],
                                                      child_codon_idxs[:, 2]]
            return child_codon_log_probs.sum()


        return optimize_branch_length(
            log_pcp_probability,
            starting_branch_length.double().item(),
            **optimization_kwargs,
        )[0]

    def find_optimal_branch_lengths(self, dataset, **optimization_kwargs):
        optimal_lengths = []

        for parent_idxs, child_idxs, rates, subs_probs, codon_mask, starting_length in tqdm(
            zip(
                dataset.nt_parents,
                dataset.nt_children,
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
