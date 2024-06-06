"""
Here we define a mutation-selection model that is just about mutation vs no mutation, and is trainable.

We'll use these conventions:

* B is the batch size
* L is the max sequence length

"""

import copy
import multiprocessing as mp

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# Amazingly, using one thread makes things 50x faster for branch length
# optimization on our server.
torch.set_num_threads(1)

import numpy as np
import pandas as pd

from tqdm import tqdm

from epam.torch_common import optimize_branch_length
from epam.models import WrappedBinaryMutSel
import epam.molevol as molevol
import epam.sequences as sequences
from epam.sequences import (
    aa_subs_indicator_tensor_of,
    translate_sequences,
)

from netam.common import (
    MAX_AMBIG_AA_IDX,
    aa_idx_tensor_of_str_ambig,
    clamp_probability,
    aa_mask_tensor_of,
    stack_heterogeneous,
)
import netam.framework as framework
from netam.hyper_burrito import HyperBurrito


class DNSMDataset(Dataset):
    def __init__(
        self,
        nt_parents: pd.Series,
        nt_children: pd.Series,
        all_rates: torch.Tensor,
        all_subs_probs: torch.Tensor,
        branch_lengths: torch.Tensor,
    ):
        self.nt_parents = nt_parents
        self.nt_children = nt_children
        self.all_rates = all_rates
        self.all_subs_probs = all_subs_probs

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
        self.aa_parents_idxs = torch.full(
            (pcp_count, self.max_aa_seq_len), MAX_AMBIG_AA_IDX
        )
        self.aa_subs_indicator_tensor = torch.zeros((pcp_count, self.max_aa_seq_len))

        self.mask = torch.ones((pcp_count, self.max_aa_seq_len), dtype=torch.bool)

        for i, (aa_parent, aa_child) in enumerate(zip(aa_parents, aa_children)):
            self.mask[i, :] = aa_mask_tensor_of(aa_parent, self.max_aa_seq_len)
            aa_seq_len = len(aa_parent)
            self.aa_parents_idxs[i, :aa_seq_len] = aa_idx_tensor_of_str_ambig(aa_parent)
            self.aa_subs_indicator_tensor[i, :aa_seq_len] = aa_subs_indicator_tensor_of(
                aa_parent, aa_child
            )

        assert torch.all(self.mask.sum(dim=1) > 0)
        assert torch.max(self.aa_parents_idxs) <= MAX_AMBIG_AA_IDX

        self._branch_lengths = branch_lengths
        self.update_neutral_aa_mut_probs()

    @classmethod
    def of_seriess(
        cls,
        nt_parents: pd.Series,
        nt_children: pd.Series,
        all_rates_series: pd.Series,
        all_subs_probs_series: pd.Series,
        branch_length_multiplier=5.0,
    ):
        """
        Alternative constructor that takes the raw data and calculates the
        initial branch lengths.

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
            stack_heterogeneous(all_rates_series.reset_index(drop=True)),
            stack_heterogeneous(all_subs_probs_series.reset_index(drop=True)),
            initial_branch_lengths,
        )

    @classmethod
    def of_pcp_df(cls, pcp_df, branch_length_multiplier=5.0):
        """
        Alternative constructor that takes in a pcp_df and calculates the
        initial branch lengths.
        """
        assert "rates" in pcp_df.columns, "pcp_df must have a neutral rates column"
        return cls.of_seriess(
            pcp_df["parent"],
            pcp_df["child"],
            pcp_df["rates"],
            pcp_df["subs_probs"],
            branch_length_multiplier=branch_length_multiplier,
        )

    def clone(self):
        """Make a deep copy of the dataset."""
        new_dataset = DNSMDataset(
            self.nt_parents,
            self.nt_children,
            self.all_rates.copy(),
            self.all_subs_probs.copy(),
            self._branch_lengths.copy(),
        )
        return new_dataset

    def subset_via_indices(self, indices):
        """
        Create a new dataset with a subset of the data, as per `indices`.

        Whether the new dataset is a deep copy or a shallow copy using slices
        depends on `indices`: if `indices` is an iterable of integers, then we
        make a deep copy, otherwise we use slices to make a shallow copy.
        """
        new_dataset = DNSMDataset(
            self.nt_parents[indices].reset_index(drop=True),
            self.nt_children[indices].reset_index(drop=True),
            self.all_rates[indices],
            self.all_subs_probs[indices],
            self._branch_lengths[indices],
        )
        return new_dataset

    def split(self, into_count: int):
        """
        Split self into a list of into_count subsets.
        """
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
        self.update_neutral_aa_mut_probs()

    def export_branch_lengths(self, out_csv_path):
        pd.DataFrame({"branch_length": self.branch_lengths}).to_csv(
            out_csv_path, index=False
        )

    def load_branch_lengths(self, in_csv_path):
        self.branch_lengths = torch.Tensor(
            pd.read_csv(in_csv_path)["branch_length"].values
        )

    def update_neutral_aa_mut_probs(self):
        neutral_aa_mut_prob_l = []

        for nt_parent, mask, rates, branch_length, subs_probs in zip(
            self.nt_parents,
            self.mask,
            self.all_rates,
            self._branch_lengths,
            self.all_subs_probs,
        ):
            mask = mask.to("cpu")
            rates = rates.to("cpu")
            subs_probs = subs_probs.to("cpu")
            # Note we are replacing all Ns with As, which means that we need to be careful
            # with masking out these positions later. We do this below.
            parent_idxs = sequences.nt_idx_tensor_of_str(nt_parent.replace("N", "A"))
            parent_len = len(nt_parent)

            mut_probs = 1.0 - torch.exp(-branch_length * rates[:parent_len])
            normed_subs_probs = molevol.normalize_sub_probs(
                parent_idxs, subs_probs[:parent_len, :]
            )

            neutral_aa_mut_prob = molevol.neutral_aa_mut_probs(
                parent_idxs.reshape(-1, 3),
                mut_probs.reshape(-1, 3),
                normed_subs_probs.reshape(-1, 3, 4),
            )

            if not torch.isfinite(neutral_aa_mut_prob).all():
                print(f"Found a non-finite neutral_aa_mut_prob")
                print(f"nt_parent: {nt_parent}")
                print(f"mask: {mask}")
                print(f"rates: {rates}")
                print(f"subs_probs: {subs_probs}")
                print(f"branch_length: {branch_length}")
                raise ValueError(
                    f"neutral_aa_mut_prob is not finite: {neutral_aa_mut_prob}"
                )

            # Ensure that all values are positive before taking the log later
            neutral_aa_mut_prob = clamp_probability(neutral_aa_mut_prob)

            pad_len = self.max_aa_seq_len - neutral_aa_mut_prob.shape[0]
            if pad_len > 0:
                neutral_aa_mut_prob = F.pad(
                    neutral_aa_mut_prob, (0, pad_len), value=1e-8
                )
            # Here we zero out masked positions.
            neutral_aa_mut_prob *= mask

            neutral_aa_mut_prob_l.append(neutral_aa_mut_prob)

        # Note that our masked out positions will have a nan log probability,
        # which will require us to handle them correctly downstream.
        self.log_neutral_aa_mut_probs = torch.log(torch.stack(neutral_aa_mut_prob_l))

    def __len__(self):
        return len(self.aa_parents_idxs)

    def __getitem__(self, idx):
        return {
            "aa_parents_idxs": self.aa_parents_idxs[idx],
            "subs_indicator": self.aa_subs_indicator_tensor[idx],
            "mask": self.mask[idx],
            "log_neutral_aa_mut_probs": self.log_neutral_aa_mut_probs[idx],
            "rates": self.all_rates[idx],
            "subs_probs": self.all_subs_probs[idx],
        }

    def to(self, device):
        self.aa_parents_idxs = self.aa_parents_idxs.to(device)
        self.aa_subs_indicator_tensor = self.aa_subs_indicator_tensor.to(device)
        self.mask = self.mask.to(device)
        self.log_neutral_aa_mut_probs = self.log_neutral_aa_mut_probs.to(device)
        self.all_rates = self.all_rates.to(device)
        self.all_subs_probs = self.all_subs_probs.to(device)


def train_test_datasets_of_pcp_df(pcp_df, train_frac=0.8, branch_length_multiplier=5.0):
    train_len = int(train_frac * len(pcp_df))
    pcp_df = pcp_df.reset_index(drop=True)
    # take the first train_len rows for train_df
    train_df = pcp_df.iloc[:train_len]
    val_df = pcp_df.iloc[train_len:]
    val_dataset = DNSMDataset.of_pcp_df(
        val_df, branch_length_multiplier=branch_length_multiplier
    )
    if train_frac == 0.0:
        return None, val_dataset
    # else:
    train_dataset = DNSMDataset.of_pcp_df(
        train_df, branch_length_multiplier=branch_length_multiplier
    )
    return train_dataset, val_dataset


class DNSMBurrito(framework.Burrito):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wrapped_model = WrappedBinaryMutSel(self.model, weights_directory=None)

    def load_branch_lengths(self, in_csv_prefix):
        if self.train_dataset is not None:
            self.train_dataset.load_branch_lengths(
                in_csv_prefix + ".train_branch_lengths.csv"
            )
        self.val_dataset.load_branch_lengths(in_csv_prefix + ".val_branch_lengths.csv")

    def predictions_of_batch(self, batch):
        """
        Make predictions for a batch of data.

        Note that we use the mask for prediction as part of the input for the
        transformer, though we don't mask the predictions themselves.
        """
        aa_parents_idxs = batch["aa_parents_idxs"].to(self.device)
        mask = batch["mask"].to(self.device)
        log_neutral_aa_mut_probs = batch["log_neutral_aa_mut_probs"].to(self.device)
        if not torch.isfinite(log_neutral_aa_mut_probs[mask]).all():
            raise ValueError(
                f"log_neutral_aa_mut_probs has non-finite values at relevant positions: {log_neutral_aa_mut_probs[mask]}"
            )
        log_selection_factors = self.model(aa_parents_idxs, mask)
        # Take the product of the neutral mutation probabilities and the selection factors.
        predictions = torch.exp(log_neutral_aa_mut_probs + log_selection_factors)
        assert torch.isfinite(predictions).all()
        predictions = clamp_probability(predictions)
        return predictions

    def loss_of_batch(self, batch):
        aa_subs_indicator = batch["subs_indicator"].to(self.device)
        mask = batch["mask"].to(self.device)
        aa_subs_indicator = aa_subs_indicator.masked_select(mask)
        predictions = self.predictions_of_batch(batch).masked_select(mask)
        return self.bce_loss(predictions, aa_subs_indicator)

    def _find_optimal_branch_length(
        self,
        parent,
        child,
        rates,
        subs_probs,
        starting_branch_length,
        **optimization_kwargs,
    ):
        if parent == child:
            return 0.0
        log_pcp_probability = self.wrapped_model._build_log_pcp_probability(
            parent, child, rates, subs_probs
        )
        if type(starting_branch_length) == torch.Tensor:
            starting_branch_length = starting_branch_length.detach().item()
        return optimize_branch_length(
            log_pcp_probability, starting_branch_length, **optimization_kwargs
        )

    def serial_find_optimal_branch_lengths(self, dataset, **optimization_kwargs):
        optimal_lengths = []
        failed_count = 0

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
            branch_length, failed_to_converge = self._find_optimal_branch_length(
                parent,
                child,
                rates[: len(parent)],
                subs_probs[: len(parent), :],
                starting_length,
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
        with mp.Pool(worker_count) as pool:
            splits = dataset.split(worker_count)
            results = pool.starmap(
                worker_optimize_branch_length,
                [(self.model, split, optimization_kwargs) for split in splits],
            )
        return torch.cat(results)

    def to_crepe(self):
        training_hyperparameters = {
            key: self.__dict__[key]
            for key in [
                "learning_rate",
            ]
        }
        encoder = framework.PlaceholderEncoder()
        return framework.Crepe(encoder, self.model, training_hyperparameters)


def worker_optimize_branch_length(model, dataset, optimization_kwargs):
    """
    The worker used for parallel branch length optimization.
    """
    burrito = DNSMBurrito(None, dataset, copy.deepcopy(model))
    return burrito.serial_find_optimal_branch_lengths(dataset, **optimization_kwargs)


class DNSMHyperBurrito(HyperBurrito):
    # Note that we have to write the args out explicitly because we use some magic to filter kwargs in the optuna_objective method.
    def burrito_of_model(
        self,
        model,
        device,
        batch_size=1024,
        learning_rate=0.1,
        min_learning_rate=1e-4,
        l2_regularization_coeff=1e-6,
    ):
        model.to(device)
        burrito = DNSMBurrito(
            self.train_dataset,
            self.val_dataset,
            model,
            batch_size=batch_size,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            l2_regularization_coeff=l2_regularization_coeff,
        )
        return burrito
