from abc import ABC, abstractmethod
import copy
import multiprocessing as mp
from functools import partial

import torch
from torch.utils.data import Dataset

# Amazingly, using one thread makes things 50x faster for branch length
# optimization on our server.
torch.set_num_threads(1)

import numpy as np
import pandas as pd

from tqdm import tqdm

from netam.common import (
    MAX_AMBIG_AA_IDX,
    aa_idx_tensor_of_str_ambig,
    aa_mask_tensor_of,
    stack_heterogeneous,
)
import netam.framework as framework
import netam.molevol as molevol
import netam.sequences as sequences
from netam.sequences import (
    aa_subs_indicator_tensor_of,
    translate_sequences,
)


class DXSMDataset(Dataset, ABC):
    def __init__(
        self,
        nt_parents: pd.Series,
        nt_children: pd.Series,
        nt_ratess: torch.Tensor,
        nt_cspss: torch.Tensor,
        branch_lengths: torch.Tensor,
        multihit_model=None,
    ):
        self.nt_parents = nt_parents
        self.nt_children = nt_children
        self.nt_ratess = nt_ratess
        self.nt_cspss = nt_cspss
        self.multihit_model = copy.deepcopy(multihit_model)
        if multihit_model is not None:
            # We want these parameters to act like fixed data. This is essential
            # for multithreaded branch length optimization to work.
            self.multihit_model.values.requires_grad_(False)

        assert len(self.nt_parents) == len(self.nt_children)
        pcp_count = len(self.nt_parents)

        for parent, child in zip(self.nt_parents, self.nt_children):
            if parent == child:
                raise ValueError(
                    f"Found an identical parent and child sequence: {parent}"
                )

        aa_parents = translate_sequences(self.nt_parents)
        aa_children = translate_sequences(self.nt_children)
        self.max_aa_seq_len = max(len(seq) for seq in aa_parents)
        # We have sequences of varying length, so we start with all tensors set
        # to the ambiguous amino acid, and then will fill in the actual values
        # below.
        self.aa_parents_idxss = torch.full(
            (pcp_count, self.max_aa_seq_len), MAX_AMBIG_AA_IDX
        )
        self.aa_children_idxss = self.aa_parents_idxss.clone()
        self.aa_subs_indicators = torch.zeros((pcp_count, self.max_aa_seq_len))

        self.masks = torch.ones((pcp_count, self.max_aa_seq_len), dtype=torch.bool)

        for i, (aa_parent, aa_child) in enumerate(zip(aa_parents, aa_children)):
            self.masks[i, :] = aa_mask_tensor_of(aa_parent, self.max_aa_seq_len)
            aa_seq_len = len(aa_parent)
            self.aa_parents_idxss[i, :aa_seq_len] = aa_idx_tensor_of_str_ambig(
                aa_parent
            )
            self.aa_children_idxss[i, :aa_seq_len] = aa_idx_tensor_of_str_ambig(
                aa_child
            )
            self.aa_subs_indicators[i, :aa_seq_len] = aa_subs_indicator_tensor_of(
                aa_parent, aa_child
            )

        assert torch.all(self.masks.sum(dim=1) > 0)
        assert torch.max(self.aa_parents_idxss) <= MAX_AMBIG_AA_IDX

        self._branch_lengths = branch_lengths
        self.update_neutral_probs()

    @classmethod
    def of_seriess(
        cls,
        nt_parents: pd.Series,
        nt_children: pd.Series,
        nt_rates_series: pd.Series,
        nt_csps_series: pd.Series,
        branch_length_multiplier=5.0,
        multihit_model=None,
    ):
        """Alternative constructor that takes the raw data and calculates the initial
        branch lengths.

        The `_series` arguments are series of Tensors which get stacked to
        create the full object.
        """
        initial_branch_lengths = np.array(
            [
                sequences.nt_mutation_frequency(parent, child)
                * branch_length_multiplier
                for parent, child in zip(nt_parents, nt_children)
            ]
        )
        return cls(
            nt_parents.reset_index(drop=True),
            nt_children.reset_index(drop=True),
            stack_heterogeneous(nt_rates_series.reset_index(drop=True)),
            stack_heterogeneous(nt_csps_series.reset_index(drop=True)),
            initial_branch_lengths,
            multihit_model=multihit_model,
        )

    @classmethod
    def of_pcp_df(cls, pcp_df, branch_length_multiplier=5.0, multihit_model=None):
        """Alternative constructor that takes in a pcp_df and calculates the initial
        branch lengths."""
        assert (
            "nt_rates" in pcp_df.columns
        ), "pcp_df must have a neutral nt_rates column"
        return cls.of_seriess(
            pcp_df["parent"],
            pcp_df["child"],
            pcp_df["nt_rates"],
            pcp_df["nt_csps"],
            branch_length_multiplier=branch_length_multiplier,
            multihit_model=multihit_model,
        )

    @classmethod
    def train_val_datasets_of_pcp_df(
        cls, pcp_df, branch_length_multiplier=5.0, multihit_model=None
    ):
        """Perform a train-val split based on the 'in_train' column.

        This is a class method so it works for subclasses.
        """
        train_df = pcp_df[pcp_df["in_train"]].reset_index(drop=True)
        val_df = pcp_df[~pcp_df["in_train"]].reset_index(drop=True)

        val_dataset = cls.of_pcp_df(
            val_df,
            branch_length_multiplier=branch_length_multiplier,
            multihit_model=multihit_model,
        )

        if len(train_df) == 0:
            return None, val_dataset
        # else:
        train_dataset = cls.of_pcp_df(
            train_df,
            branch_length_multiplier=branch_length_multiplier,
            multihit_model=multihit_model,
        )

        return train_dataset, val_dataset

    def clone(self):
        """Make a deep copy of the dataset."""
        new_dataset = self.__class__(
            self.nt_parents,
            self.nt_children,
            self.nt_ratess.copy(),
            self.nt_cspss.copy(),
            self._branch_lengths.copy(),
            multihit_model=self.multihit_model,
        )
        return new_dataset

    def subset_via_indices(self, indices):
        """Create a new dataset with a subset of the data, as per `indices`.

        Whether the new dataset is a deep copy or a shallow copy using slices
        depends on `indices`: if `indices` is an iterable of integers, then we
        make a deep copy, otherwise we use slices to make a shallow copy.
        """
        new_dataset = self.__class__(
            self.nt_parents[indices].reset_index(drop=True),
            self.nt_children[indices].reset_index(drop=True),
            self.nt_ratess[indices],
            self.nt_cspss[indices],
            self._branch_lengths[indices],
            multihit_model=self.multihit_model,
        )
        return new_dataset

    def split(self, into_count: int):
        """Split self into a list of into_count subsets."""
        dataset_size = len(self)
        indices = list(range(dataset_size))
        split_indices = np.array_split(indices, into_count)
        subsets = [self.subset_via_indices(split_indices[i]) for i in range(into_count)]
        return subsets

    @property
    def branch_lengths(self):
        return self._branch_lengths

    @branch_lengths.setter
    def branch_lengths(self, new_branch_lengths):
        assert len(new_branch_lengths) == len(self._branch_lengths), (
            f"Expected {len(self._branch_lengths)} branch lengths, "
            f"got {len(new_branch_lengths)}"
        )
        assert torch.all(torch.isfinite(new_branch_lengths) & (new_branch_lengths > 0))
        self._branch_lengths = new_branch_lengths
        self.update_neutral_probs()

    def __len__(self):
        return len(self.aa_parents_idxss)

    def export_branch_lengths(self, out_csv_path):
        pd.DataFrame({"branch_length": self.branch_lengths}).to_csv(
            out_csv_path, index=False
        )

    def load_branch_lengths(self, in_csv_path):
        self.branch_lengths = torch.Tensor(
            pd.read_csv(in_csv_path)["branch_length"].values
        )

    @abstractmethod
    def update_neutral_probs(self):
        pass


class DXSMBurrito(framework.Burrito, ABC):
    def _find_optimal_branch_length(
        self,
        parent,
        child,
        nt_rates,
        nt_csps,
        starting_branch_length,
        multihit_model,
        **optimization_kwargs,
    ):
        if parent == child:
            return 0.0
        sel_matrix = self.build_selection_matrix_from_parent(parent)
        log_pcp_probability = molevol.mutsel_log_pcp_probability_of(
            sel_matrix, parent, child, nt_rates, nt_csps, multihit_model
        )
        if isinstance(starting_branch_length, torch.Tensor):
            starting_branch_length = starting_branch_length.detach().item()
        return molevol.optimize_branch_length(
            log_pcp_probability, starting_branch_length, **optimization_kwargs
        )

    def serial_find_optimal_branch_lengths(self, dataset, **optimization_kwargs):
        optimal_lengths = []
        failed_count = 0

        for parent, child, nt_rates, nt_csps, starting_length in tqdm(
            zip(
                dataset.nt_parents,
                dataset.nt_children,
                dataset.nt_ratess,
                dataset.nt_cspss,
                dataset.branch_lengths,
            ),
            total=len(dataset.nt_parents),
            desc="Finding optimal branch lengths",
        ):
            branch_length, failed_to_converge = self._find_optimal_branch_length(
                parent,
                child,
                nt_rates[: len(parent)],
                nt_csps[: len(parent), :],
                starting_length,
                dataset.multihit_model,
                **optimization_kwargs,
            )

            optimal_lengths.append(branch_length)
            failed_count += failed_to_converge

        if failed_count > 0:
            print(
                f"Branch length optimization failed to converge for {failed_count} of {len(dataset)} sequences."
            )

        return torch.tensor(optimal_lengths)

    def find_optimal_branch_lengths(self, dataset, **optimization_kwargs):
        worker_count = min(mp.cpu_count() // 2, 10)
        # # The following can be used when one wants a better traceback.
        # burrito = self.__class__(None, dataset, copy.deepcopy(self.model))
        # return burrito.serial_find_optimal_branch_lengths(dataset, **optimization_kwargs)
        our_optimize_branch_length = partial(
            worker_optimize_branch_length,
            self.__class__,
        )
        with mp.Pool(worker_count) as pool:
            splits = dataset.split(worker_count)
            results = pool.starmap(
                our_optimize_branch_length,
                [(self.model, split, optimization_kwargs) for split in splits],
            )
        return torch.cat(results)

    def to_crepe(self):
        training_hyperparameters = {
            key: self.__dict__[key]
            for key in [
                "optimizer_name",
                "batch_size",
                "learning_rate",
                "min_learning_rate",
                "weight_decay",
            ]
        }
        encoder = framework.PlaceholderEncoder()
        return framework.Crepe(encoder, self.model, training_hyperparameters)

    @abstractmethod
    def loss_of_batch(self, batch):
        pass


def worker_optimize_branch_length(burrito_class, model, dataset, optimization_kwargs):
    """The worker used for parallel branch length optimization."""
    burrito = burrito_class(None, dataset, copy.deepcopy(model))
    return burrito.serial_find_optimal_branch_lengths(dataset, **optimization_kwargs)
