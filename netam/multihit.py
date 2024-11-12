"""Burrito and Dataset classes for training a model to predict simple hit class
corrections to codon probabilities.

Each codon mutation is hit class 0, 1, 2, or 3, corresponding to 0, 1, 2, or 3 mutations
in the codon.

The hit class corrections are three scalar values, one for each nonzero hit class. To
apply the correction to existing codon probability predictions, we multiply the
probability of each child codon by the correction factor for its hit class, then
renormalize. The correction factor for hit class 0 is fixed at 1.
"""

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
from typing import Sequence, List, Tuple

from netam.molevol import (
    reshape_for_codons,
    optimize_branch_length,
    codon_probs_of_parent_scaled_nt_rates_and_csps,
)
from netam.hit_class import hit_class_probs_tensor
from netam import sequences
from netam.common import stack_heterogeneous
import netam.framework as framework
from netam.framework import Burrito
from netam.models import HitClassModel


def _trim_to_codon_boundary_and_max_len(
    seqs: List[Sequence], max_len: int = None
) -> List[Sequence]:
    """Trims sequences to codon boundary and maximum length.

    No assumption is made about the data of a sequence, other than that it is
    indexable (string or list of nucleotide indices both work).

    `max_len` is the maximum number of nucleotides to be preserved.
    If `max_len` is None, does not enforce a maximum length.
    """
    if max_len is None:
        return [seq[: len(seq) - len(seq) % 3] for seq in seqs]
    else:
        max_codon_len = max_len - max_len % 3
        return [seq[: min(len(seq) - len(seq) % 3, max_codon_len)] for seq in seqs]


def _observed_hit_classes(parents: Sequence[str], children: Sequence[str]):
    """Compute the observed hit classes between parent and child sequences.

    Args:
        parents (Sequence[str]): A list of parent sequences.
        children (Sequence[str]): A list of the corresponding child sequences.

    Returns:
        List[torch.Tensor]: A list of tensors, each containing the observed
            hit classes for each codon in the parent sequence. At any codon position
            where the parent or child sequence contains an N, the corresponding tensor
            element will be -100.
    """
    labels = []
    for parent_seq, child_seq in zip(parents, children):

        assert len(parent_seq) == len(child_seq)
        codon_count = len(parent_seq) // 3
        valid_length = codon_count * 3

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
        )  # Pad with -100s

        # Update the labels tensor for this row
        labels.append(torch.tensor(padded_mutations, dtype=torch.int))
    return labels


class HitClassDataset(Dataset):
    def __init__(
        self,
        nt_parents: Sequence[str],
        nt_children: Sequence[str],
        nt_ratess: Sequence[List[float]],
        nt_cspss: Sequence[List[List[float]]],
        branch_length_multiplier: float = 1.0,
    ):
        trimmed_parents = _trim_to_codon_boundary_and_max_len(nt_parents)
        trimmed_children = _trim_to_codon_boundary_and_max_len(nt_children)
        self.nt_parents = stack_heterogeneous(
            pd.Series(
                sequences.nt_idx_tensor_of_str(parent.replace("N", "A"))
                for parent in trimmed_parents
            )
        )
        self.nt_children = stack_heterogeneous(
            pd.Series(
                sequences.nt_idx_tensor_of_str(child.replace("N", "A"))
                for child in trimmed_children
            )
        )
        self.nt_ratess = stack_heterogeneous(
            pd.Series(_trim_to_codon_boundary_and_max_len(nt_ratess)).reset_index(
                drop=True
            )
        )
        self.nt_cspss = stack_heterogeneous(
            pd.Series(_trim_to_codon_boundary_and_max_len(nt_cspss)).reset_index(
                drop=True
            )
        )

        assert len(self.nt_parents) == len(self.nt_children)

        for parent, child in zip(trimmed_parents, trimmed_children):
            if parent == child:
                raise ValueError(
                    f"Found an identical parent and child sequence: {parent}"
                )
            assert len(parent) == len(child)

        self.observed_hcs = stack_heterogeneous(
            _observed_hit_classes(trimmed_parents, trimmed_children), pad_value=-100
        ).long()

        # This masks any codon that has an N in the parent or child. This may
        # change with https://github.com/matsengrp/netam/issues/16
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
        """Compute hit class probabilities for all codons in each sequence based on
        current branch lengths."""
        new_codon_probs = []
        new_hc_probs = []
        for (
            encoded_parent,
            nt_rates,
            nt_csps,
            branch_length,
        ) in zip(
            self.nt_parents,
            self.nt_ratess,
            self.nt_cspss,
            self.branch_lengths,
        ):
            scaled_rates = branch_length * nt_rates

            codon_probs = codon_probs_of_parent_scaled_nt_rates_and_csps(
                encoded_parent,
                scaled_rates[: len(encoded_parent)],
                nt_csps[: len(encoded_parent)],
            )
            new_codon_probs.append(codon_probs)

            new_hc_probs.append(
                hit_class_probs_tensor(reshape_for_codons(encoded_parent), codon_probs)
            )
        self.codon_probs = torch.stack(new_codon_probs)
        self.hit_class_probs = torch.stack(new_hc_probs)

    # A couple of these methods could maybe be moved to a super class, which itself subclasses Dataset
    def export_branch_lengths(self, out_csv_path):
        pd.DataFrame({"branch_length": self.branch_lengths}).to_csv(
            out_csv_path, index=False
        )

    def load_branch_lengths(self, in_csv_path):
        self.branch_lengths = torch.tensor(
            pd.read_csv(in_csv_path)["branch_length"].values
        )

    def __len__(self):
        return len(self.nt_parents)

    def __getitem__(self, idx):
        return {
            "parent": self.nt_parents[idx],
            "child": self.nt_children[idx],
            "observed_hcs": self.observed_hcs[idx],
            "nt_rates": self.nt_ratess[idx],
            "nt_csps": self.nt_cspss[idx],
            "hit_class_probs": self.hit_class_probs[idx],
            "codon_probs": self.codon_probs[idx],
            "codon_mask": self.codon_mask[idx],
        }

    def to(self, device):
        self.nt_parents = self.nt_parents.to(device)
        self.nt_children = self.nt_children.to(device)
        self.observed_hcs = self.observed_hcs.to(device)
        self.nt_ratess = self.nt_ratess.to(device)
        self.nt_cspss = self.nt_cspss.to(device)
        self.hit_class_probs = self.hit_class_probs.to(device)
        self.codon_mask = self.codon_mask.to(device)
        self.branch_lengths = self.branch_lengths.to(device)


def flatten_and_mask_sequence_codons(
    input_tensor: torch.Tensor, codon_mask: torch.Tensor = None
):
    """Flatten first dimension of input_tensor, applying codon_mask first if provided.

    This is useful for input_tensors whose first dimension represents sequences, and
    whose second dimension represents codons. The resulting tensor will then aggregate
    the codons of all sequences into the first dimension.
    """
    flat_input = input_tensor.flatten(start_dim=0, end_dim=1)
    if codon_mask is not None:
        flat_codon_mask = codon_mask.flatten()
        flat_input = flat_input[flat_codon_mask]
    return flat_input


def child_codon_probs_from_per_parent_probs(per_parent_probs, child_codon_idxs):
    """Calculate the probability of each child codon given the parent codon
    probabilities.

    Args:
        per_parent_probs (torch.Tensor): A (codon_count, 4, 4, 4) shaped tensor containing the probabilities
            of each possible target codon, for each parent codon.
        child_codon_idxs (torch.Tensor): A (codon_count, 3) shaped tensor containing the nucleotide indices for each child codon.

    Returns:
        torch.Tensor: A (codon_count,) shaped tensor containing the probabilities of each child codon.
    """
    return per_parent_probs[
        torch.arange(child_codon_idxs.size(0)),
        child_codon_idxs[:, 0],
        child_codon_idxs[:, 1],
        child_codon_idxs[:, 2],
    ]


def child_codon_probs_corrected(
    uncorrected_per_parent_probs, parent_codon_idxs, child_codon_idxs, model
):
    """Calculate the probability of each child codon given the parent codon
    probabilities, corrected by hit class factors.

    Args:
        uncorrected_per_parent_probs (torch.Tensor): A (codon_count, 4, 4, 4) shaped tensor containing the probabilities
            of each possible target codon, for each parent codon.
        parent_codon_idxs (torch.Tensor): A (codon_count, 3) shaped tensor containing the nucleotide indices for each parent codon
        child_codon_idxs (torch.Tensor): A (codon_count, 3) shaped tensor containing the nucleotide indices for each child codon
        model: A HitClassModel implementing the desired correction.

    Returns:
        torch.Tensor: A (codon_count,) shaped tensor containing the corrected probabilities of each child codon.
    """

    corrected_per_parent_probs = model(parent_codon_idxs, uncorrected_per_parent_probs)
    return child_codon_probs_from_per_parent_probs(
        corrected_per_parent_probs, child_codon_idxs
    )


class MultihitBurrito(Burrito):
    def __init__(
        self,
        train_dataset: HitClassDataset,
        val_dataset: HitClassDataset,
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

        # remove first dimension of parent_idxs by concatenating all of its elements
        codon_mask_flat = codon_mask.flatten(start_dim=0, end_dim=1)
        parent_codons_flat = reshape_for_codons(
            parent_idxs.flatten(start_dim=0, end_dim=1)
        )[codon_mask_flat]
        child_codons_flat = reshape_for_codons(
            child_idxs.flatten(start_dim=0, end_dim=1)
        )[codon_mask_flat]

        flat_masked_codon_probs = flatten_and_mask_sequence_codons(
            codon_probs, codon_mask=codon_mask
        )

        child_codon_logprobs = child_codon_probs_corrected(
            flat_masked_codon_probs,
            parent_codons_flat,
            child_codons_flat,
            self.model,
        ).log()
        return -child_codon_logprobs.sum()

    def _find_optimal_branch_length(
        self,
        parent_idxs,
        child_idxs,
        nt_rates,
        nt_csps,
        codon_mask,
        starting_branch_length,
        **optimization_kwargs,
    ):

        def log_pcp_probability(log_branch_length):
            # We want to first return the log-probability of the observed branch, using codon probs.
            # Then we'll want to adjust codon probs using our hit class probabilities
            branch_length = torch.exp(log_branch_length)
            scaled_rates = nt_rates * branch_length
            # Rates is a 1d tensor containing one rate for each nt site.

            # Codon probs is a Cx4x4x4 tensor containing for each codon idx the
            # distribution on possible target codons (all 64 of them!)
            codon_probs = codon_probs_of_parent_scaled_nt_rates_and_csps(
                parent_idxs, scaled_rates, nt_csps
            )[codon_mask]

            child_codon_idxs = reshape_for_codons(child_idxs)[codon_mask]
            parent_codon_idxs = reshape_for_codons(parent_idxs)[codon_mask]
            return (
                child_codon_probs_corrected(
                    codon_probs, parent_codon_idxs, child_codon_idxs, self.model
                )
                .log()
                .sum()
            )

        return optimize_branch_length(
            log_pcp_probability,
            starting_branch_length.double().item(),
            **optimization_kwargs,
        )[0]

    def find_optimal_branch_lengths(self, dataset, **optimization_kwargs):
        optimal_lengths = []

        for (
            parent_idxs,
            child_idxs,
            nt_rates,
            nt_csps,
            codon_mask,
            starting_length,
        ) in tqdm(
            zip(
                dataset.nt_parents,
                dataset.nt_children,
                dataset.nt_ratess,
                dataset.nt_cspss,
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
                    nt_rates[: len(parent_idxs)],
                    nt_csps[: len(parent_idxs), :],
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


def hit_class_dataset_from_pcp_df(
    pcp_df: pd.DataFrame, branch_length_multiplier: int = 1.0
) -> HitClassDataset:
    nt_parents = pcp_df["parent"].reset_index(drop=True)
    nt_children = pcp_df["child"].reset_index(drop=True)
    nt_rates = pcp_df["nt_rates"].reset_index(drop=True)
    nt_csps = pcp_df["nt_csps"].reset_index(drop=True)

    return HitClassDataset(
        nt_parents,
        nt_children,
        nt_rates,
        nt_csps,
        branch_length_multiplier=branch_length_multiplier,
    )


def train_test_datasets_of_pcp_df(
    pcp_df: pd.DataFrame, train_frac: float = 0.8, branch_length_multiplier: float = 1.0
) -> Tuple[HitClassDataset, HitClassDataset]:
    """Splits a pcp_df prepared by `prepare_pcp_df` into a training and testing
    HitClassDataset."""
    nt_parents = pcp_df["parent"].reset_index(drop=True)
    nt_children = pcp_df["child"].reset_index(drop=True)
    nt_rates = pcp_df["nt_rates"].reset_index(drop=True)
    nt_csps = pcp_df["nt_csps"].reset_index(drop=True)

    train_len = int(train_frac * len(nt_parents))
    train_parents, val_parents = nt_parents[:train_len], nt_parents[train_len:]
    train_children, val_children = nt_children[:train_len], nt_children[train_len:]
    train_rates, val_rates = nt_rates[:train_len], nt_rates[train_len:]
    train_nt_csps, val_nt_csps = (
        nt_csps[:train_len],
        nt_csps[train_len:],
    )
    val_dataset = HitClassDataset(
        val_parents,
        val_children,
        val_rates,
        val_nt_csps,
        branch_length_multiplier=branch_length_multiplier,
    )
    if train_frac == 0.0:
        return None, val_dataset
    # else:
    train_dataset = HitClassDataset(
        train_parents,
        train_children,
        train_rates,
        train_nt_csps,
        branch_length_multiplier=branch_length_multiplier,
    )
    return val_dataset, train_dataset


def prepare_pcp_df(
    pcp_df: pd.DataFrame, crepe: framework.Crepe, site_count: int
) -> pd.DataFrame:
    """Trim parent and child sequences in pcp_df to codon boundaries and add the
    nt_rates and substitution probabilities.

    Returns the modified dataframe, which is the input dataframe modified in-place.
    """
    pcp_df["parent"] = _trim_to_codon_boundary_and_max_len(pcp_df["parent"], site_count)
    pcp_df["child"] = _trim_to_codon_boundary_and_max_len(pcp_df["child"], site_count)
    pcp_df = pcp_df[pcp_df["parent"] != pcp_df["child"]].reset_index(drop=True)
    ratess, cspss = framework.trimmed_shm_model_outputs_of_crepe(
        crepe, pcp_df["parent"]
    )
    pcp_df["nt_rates"] = ratess
    pcp_df["nt_csps"] = cspss
    return pcp_df
