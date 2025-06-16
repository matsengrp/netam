from warnings import warn
from abc import ABC, abstractmethod
import copy
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from netam.common import (
    stack_heterogeneous,
    zap_predictions_along_diagonal,
)
from netam.pretrained import name_and_multihit_model_match
import netam.framework as framework
import netam.molevol as molevol
from netam.sequences import (
    aa_idx_tensor_of_str_ambig,
    codon_mask_tensor_of,
    assert_pcp_valid,
    aa_subs_indicator_tensor_of,
    translate_sequences,
    apply_aa_mask_to_nt_sequence,
    nt_mutation_frequency,
    dataset_inputs_of_pcp_df,
    token_mask_of_aa_idxs,
    MAX_AA_TOKEN_IDX,
    RESERVED_TOKEN_REGEX,
    AA_PADDING_TOKEN,
)

# Amazingly, using one thread makes things 50x faster for branch length
# optimization on our server.
torch.set_num_threads(1)

# Enable cuDNN autotuner for better GPU performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class DXSMDataset(framework.BranchLengthDataset, ABC):
    # Not defining model_type here; instead defining it in subclasses.
    # This will raise an error if we aren't using a subclass.

    def __init__(
        self,
        nt_parents: pd.Series,
        nt_children: pd.Series,
        nt_ratess: torch.Tensor,
        nt_cspss: torch.Tensor,
        branch_lengths: torch.Tensor,
        aa_parents_idxss: torch.Tensor,
        aa_children_idxss: torch.Tensor,
        masks: torch.Tensor,
        aa_subs_indicators: torch.Tensor,
        model_known_token_count: int,
        multihit_model=None,
    ):
        # This is no longer needed here, but it seems like we should be able to verify what model version an instance is built for anyway:
        self.model_known_token_count = model_known_token_count

        self.nt_parents = nt_parents
        self.nt_children = nt_children
        self.nt_ratess = nt_ratess
        self.nt_cspss = nt_cspss
        self.max_aa_seq_len = aa_parents_idxss.shape[1]
        self.aa_parents_idxss = aa_parents_idxss
        self.aa_children_idxss = aa_children_idxss
        self.masks = masks
        self.aa_subs_indicators = aa_subs_indicators
        self.multihit_model = copy.deepcopy(multihit_model)
        if self.multihit_model is not None:
            # We want these parameters to act like fixed data. This is essential
            # for multithreaded branch length optimization to work.
            self.multihit_model = self.multihit_model.to("cpu")
            self.multihit_model.values.requires_grad_(False)

        assert len(self.nt_parents) == len(self.nt_children)

        assert self.masks.shape == self.aa_parents_idxss.shape
        assert self.masks.shape[1] * 3 == self.nt_ratess.shape[1]
        assert self.masks.shape[1] * 3 == self.nt_cspss.shape[1]
        assert torch.all(self.masks.sum(dim=1) > 0)
        assert torch.max(self.aa_parents_idxss) <= MAX_AA_TOKEN_IDX
        assert torch.max(self.aa_children_idxss) <= MAX_AA_TOKEN_IDX

        self._branch_lengths = branch_lengths
        self.update_neutral_probs()

    def __post_init__(self):
        self.move_data_to_device("cpu")

    @abstractmethod
    def move_data_to_device(self, device):
        """Move all tensors stored by the dataset to the given device."""
        pass

    @classmethod
    def of_seriess(
        cls,
        nt_parents: pd.Series,
        nt_children: pd.Series,
        nt_rates_series: pd.Series,
        nt_csps_series: pd.Series,
        model_known_token_count,
        branch_length_multiplier=5.0,
        multihit_model=None,
    ):
        """Alternative constructor that takes the raw data and calculates the initial
        branch lengths and masks.

        The `_series` arguments are series of Tensors which get stacked to
        create the full object.
        """
        initial_branch_lengths = np.array(
            [
                nt_mutation_frequency(parent, child) * branch_length_multiplier
                for parent, child in zip(nt_parents, nt_children)
            ]
        )
        pcp_count = len(nt_parents)

        nt_ratess = stack_heterogeneous(nt_rates_series.reset_index(drop=True))
        nt_cspss = stack_heterogeneous(nt_csps_series.reset_index(drop=True))
        aa_parents = translate_sequences(nt_parents)
        aa_children = translate_sequences(nt_children)
        max_aa_seq_len = max(len(seq) for seq in aa_parents)
        assert nt_ratess.shape[1] == nt_cspss.shape[1]
        rates_len = nt_ratess.shape[1] // 3
        if rates_len < max_aa_seq_len:
            raise ValueError(
                f"Expected nt_ratess to have at least {max_aa_seq_len} codons"
            )
        else:
            max_aa_seq_len = rates_len
        # We have sequences of varying length, so we start with all tensors set
        # to the ambiguous amino acid, and then will fill in the actual values
        # below.
        aa_parents_idxss = torch.full((pcp_count, max_aa_seq_len), AA_PADDING_TOKEN)
        aa_children_idxss = aa_parents_idxss.clone()
        aa_subs_indicators = torch.zeros((pcp_count, max_aa_seq_len))

        masks = torch.ones((pcp_count, max_aa_seq_len), dtype=torch.bool)

        for i, (aa_parent, aa_child) in enumerate(zip(aa_parents, aa_children)):
            masks[i, :] = codon_mask_tensor_of(
                nt_parents[i], nt_children[i], aa_length=max_aa_seq_len
            )
            aa_seq_len = len(aa_parent)
            assert_pcp_valid(
                nt_parents[i], nt_children[i], aa_mask=masks[i][:aa_seq_len]
            )

            aa_parents_idxss[i, :aa_seq_len] = aa_idx_tensor_of_str_ambig(aa_parent)
            aa_children_idxss[i, :aa_seq_len] = aa_idx_tensor_of_str_ambig(aa_child)
            aa_subs_indicators[i, :aa_seq_len] = aa_subs_indicator_tensor_of(
                aa_parent, aa_child
            )
        # # We will replace reserved tokens with Ns but use the unmodified
        # # originals for translation and mask creation.
        # Important to use the unmodified versions of nt_parents and
        # nt_children above this so they still contain special tokens.
        nt_parents = nt_parents.str.replace(RESERVED_TOKEN_REGEX, "N", regex=True)
        nt_children = nt_children.str.replace(RESERVED_TOKEN_REGEX, "N", regex=True)
        return cls(
            nt_parents.reset_index(drop=True),
            nt_children.reset_index(drop=True),
            nt_ratess,
            nt_cspss,
            initial_branch_lengths,
            aa_parents_idxss,
            aa_children_idxss,
            masks,
            aa_subs_indicators,
            model_known_token_count,
            multihit_model=multihit_model,
        )

    @classmethod
    def of_pcp_df(
        cls,
        pcp_df,
        model_known_token_count,
        branch_length_multiplier=5.0,
        multihit_model=None,
    ):
        """Alternative constructor that takes in a pcp_df and calculates the initial
        branch lengths."""
        assert (
            "nt_rates_light" in pcp_df.columns
        ), "pcp_df must have a neutral nt_rates column"
        # use sequences.prepare_heavy_light_pair and the resulting
        # added_indices to get the parent and child sequences and neutral model
        # outputs

        return cls.of_seriess(
            *dataset_inputs_of_pcp_df(pcp_df, model_known_token_count),
            model_known_token_count,
            branch_length_multiplier=branch_length_multiplier,
            multihit_model=multihit_model,
        )

    @classmethod
    def train_val_datasets_of_pcp_df(
        cls,
        pcp_df,
        model_known_token_count,
        branch_length_multiplier=5.0,
        multihit_model=None,
    ):
        """Perform a train-val split based on the 'in_train' column.

        This is a class method so it works for subclasses.
        """
        train_df = pcp_df[pcp_df["in_train"]].reset_index(drop=True)
        val_df = pcp_df[~pcp_df["in_train"]].reset_index(drop=True)

        val_dataset = cls.of_pcp_df(
            val_df,
            model_known_token_count,
            branch_length_multiplier=branch_length_multiplier,
            multihit_model=multihit_model,
        )

        if len(train_df) == 0:
            return None, val_dataset
        # else:
        train_dataset = cls.of_pcp_df(
            train_df,
            model_known_token_count,
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
            self.branch_lengths.copy(),
            self.aa_parents_idxss.copy(),
            self.aa_children_idxss.copy(),
            self.masks.copy(),
            self.aa_subs_indicators.copy(),
            self.model_known_token_count,
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
            self.branch_lengths[indices],
            self.aa_parents_idxss[indices],
            self.aa_children_idxss[indices],
            self.masks[indices],
            self.aa_subs_indicators[indices],
            self.model_known_token_count,
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

    def to(self, device):
        self.device = device

    @abstractmethod
    def update_neutral_probs(self):
        pass


class DXSMBurrito(framework.Burrito, ABC):
    # Not defining model_type here; instead defining it in subclasses.
    # This will raise an error if we aren't using a subclass.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Does model metadata match dataset?

        # For backward compatibility -- it's not possible to determine what an
        # old crepe is from its metadata :(
        if self.model.model_type != "unknown":
            if not (self.model.model_type == self.model_type):
                warn(
                    f"Model type {self.model.model_type} does not match expected type {self.model_type}. "
                    "To avoid this warning, provide `model_type` argument to model constructor."
                )
        else:
            warn(
                "Model type is unknown. This is likely an old model that does not include "
                "its type (dnsm, ddsm, or dasm, etc.) in its metadata. Be sure the model "
                "type matches the Dataset and Burrito type."
            )

        multihit_model_name = self.model.hyperparameters["multihit_model_name"]
        if not name_and_multihit_model_match(
            multihit_model_name,
            self.val_dataset.multihit_model,
        ):
            warn(
                "Validation dataset multihit model does not match the one referenced in "
                f"provided model metadata: '{multihit_model_name}'. "
                "To fix this, provide the `multihit_model_name` argument to the model "
                "constructor, or provide the corresponding multihit model instance to the Dataset constructor."
            )
        if self.train_dataset is not None:
            if not name_and_multihit_model_match(
                multihit_model_name,
                self.train_dataset.multihit_model,
            ):
                warn(
                    "Training dataset multihit model does not match the one referenced in "
                    f"provided model metadata: '{multihit_model_name}'. "
                    "To fix this, provide the `multihit_model_name` argument to the model "
                    "constructor, or provide the corresponding multihit model instance to the Dataset constructor."
                )

    def selection_factors_of_aa_idxs(self, aa_idxs, aa_mask):
        """Get the log selection factors for a batch of amino acid indices.

        aa_idxs and aa_mask are expected to be as prepared in the Dataset constructor.
        """
        # We need the model to see special tokens here. For every other purpose
        # they are masked out.
        keep_token_mask = aa_mask | token_mask_of_aa_idxs(aa_idxs)
        return self.model(aa_idxs, keep_token_mask)

    def _find_optimal_branch_length(
        self,
        parent,
        child,
        nt_rates,
        nt_csps,
        aa_mask,
        aa_parents_indices,
        starting_branch_length,
        multihit_model,
        **optimization_kwargs,
    ):
        sel_matrix = self.build_selection_matrix_from_parent_aa(
            aa_parents_indices, aa_mask
        )
        # This is essential so that it is not interpreted as indices!!
        assert aa_mask.dtype == torch.bool
        # Masks may be padded at end to account for sequences of different
        # lengths. The first part of the mask should be
        # all the valid bits for the sequence. apply_aa_mask_to_nt_sequence
        # ignores extra mask on the end.
        log_pcp_probability = molevol.mutsel_log_pcp_probability_of(
            sel_matrix[aa_mask],
            apply_aa_mask_to_nt_sequence(parent, aa_mask),
            apply_aa_mask_to_nt_sequence(child, aa_mask),
            nt_rates[aa_mask.repeat_interleave(3)],
            nt_csps[aa_mask.repeat_interleave(3)],
            multihit_model,
        )
        if isinstance(starting_branch_length, torch.Tensor):
            starting_branch_length = starting_branch_length.detach().item()
        return molevol.optimize_branch_length(
            log_pcp_probability, starting_branch_length, **optimization_kwargs
        )

    def serial_find_optimal_branch_lengths(self, dataset, **optimization_kwargs):
        optimal_lengths = []
        failed_count = 0

        for (
            parent,
            child,
            nt_rates,
            nt_csps,
            aa_mask,
            aa_parents_indices,
            starting_length,
        ) in tqdm(
            zip(
                dataset.nt_parents,
                dataset.nt_children,
                dataset.nt_ratess,
                dataset.nt_cspss,
                dataset.masks,
                dataset.aa_parents_idxss,
                dataset.branch_lengths,
            ),
            total=len(dataset.nt_parents),
            desc="Finding optimal branch lengths",
        ):
            branch_length, failed_to_converge = self._find_optimal_branch_length(
                parent,
                child,
                nt_rates,
                nt_csps,
                aa_mask,
                aa_parents_indices,
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

    def load_branch_lengths(self, in_csv_prefix):
        if self.train_dataset is not None:
            self.train_dataset.load_branch_lengths(
                in_csv_prefix + ".train_branch_lengths.csv"
            )
        self.val_dataset.load_branch_lengths(in_csv_prefix + ".val_branch_lengths.csv")

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

    # This is overridden in DNSMBurrito
    def build_selection_matrix_from_parent_aa(
        self, aa_parent_idxs: torch.Tensor, mask: torch.Tensor
    ):
        """Build a selection matrix from a single parent amino acid sequence. Inputs are
        expected to be as prepared in the Dataset constructor.

        Values at ambiguous sites are meaningless.
        """
        with torch.no_grad():
            per_aa_selection_factors = self.selection_factors_of_aa_idxs(
                aa_parent_idxs.unsqueeze(0), mask.unsqueeze(0)
            ).exp()
        return zap_predictions_along_diagonal(
            per_aa_selection_factors, aa_parent_idxs.unsqueeze(0), fill=1.0
        ).squeeze(0)

    @abstractmethod
    def loss_of_batch(self, batch):
        pass
