"""Dataset class for whichmut training with precomputed neutral rates.

This module provides the WhichmutCodonDataset class for storing and managing codon-level
parent/child sequences with precomputed neutral rates for codon-level selection
modeling.
"""

import torch
import pandas as pd
from typing import Dict, Any, Optional
from tqdm import tqdm

from netam.codon_table import (
    CODON_SINGLE_MUTATIONS,
    encode_codon_mutations,
    create_codon_masks,
)
from netam.sequences import (
    translate_sequences,
    aa_idx_tensor_of_str_ambig,
    CODONS,
    BASES_AND_N_TO_INDEX,
)


def get_sparse_neutral_rate(
    sparse_data, seq_idx, codon_pos, parent_codon_idx, child_codon_idx
):
    """Look up neutral rate in sparse format.

    Args:
        sparse_data: Dict with 'indices', 'values', 'n_mutations'
        seq_idx: Sequence index
        codon_pos: Codon position
        parent_codon_idx: Parent codon index
        child_codon_idx: Child codon index

    Returns:
        Neutral rate for the specified transition, or 0.0 if not found
    """
    indices = sparse_data["indices"][seq_idx, codon_pos]  # Shape: (max_mutations, 2)
    values = sparse_data["values"][seq_idx, codon_pos]  # Shape: (max_mutations,)
    n_mutations = sparse_data["n_mutations"][seq_idx, codon_pos].item()

    # Search for matching parent->child transition in the sparse data
    for i in range(n_mutations):
        if indices[i, 0] == parent_codon_idx and indices[i, 1] == child_codon_idx:
            return values[i]

    # Not found - return 0 (this might indicate an error in sparse encoding)
    return torch.tensor(0.0, device=values.device, dtype=values.dtype)


class WhichmutCodonDataset:
    """Dataset for whichmut training using precomputed neutral rates.

    Stores codon-level parent/child sequences, precomputed neutral rates λ_{j,c->c'},
    and mutation indicators. Designed for codon-level selection modeling.

    Does not enforce single mutation per PCP - handles arbitrary numbers of mutations.

    Supports both dense and sparse neutral rates formats for memory efficiency.
    """

    def __init__(
        self,
        nt_parents: pd.Series,  # Parent nucleotide sequences
        nt_children: pd.Series,  # Child nucleotide sequences
        codon_parents_idxss: torch.Tensor,  # Parent codons as indices (N, L_codon)
        codon_children_idxss: torch.Tensor,  # Child codons as indices (N, L_codon)
        neutral_rates_tensor: torch.Tensor,  # Precomputed λ_{j,c->c'} (N, L_codon, 65, 65) or sparse
        aa_parents_idxss: torch.Tensor,  # Parent AAs as indices (N, L_aa)
        aa_children_idxss: torch.Tensor,  # Child AAs as indices (N, L_aa)
        codon_mutation_indicators: torch.Tensor,  # Which codon sites mutated (N, L_codon)
        masks: torch.Tensor,  # Valid codon positions (N, L_codon)
        model_known_token_count: int,
        sparse_neutral_rates: Optional[Dict[str, torch.Tensor]] = None,  # Sparse format
    ):
        """Initialize WhichmutCodonDataset with precomputed neutral rates.

        All tensors should be on the same device. Neutral rates λ_{j,c->c'} are
        precomputed using viral neutral models and passed in.

        Args:
            sparse_neutral_rates: Optional dict with keys:
                - 'indices': torch.LongTensor of shape (N, L_codon, max_mutations, 2)
                  Last dim is [parent_codon_idx, child_codon_idx]
                - 'values': torch.Tensor of shape (N, L_codon, max_mutations)
                - 'n_mutations': torch.LongTensor of shape (N, L_codon) - actual mutations per position
        """
        self.nt_parents = nt_parents
        self.nt_children = nt_children
        self.codon_parents_idxss = codon_parents_idxss
        self.codon_children_idxss = codon_children_idxss
        self.neutral_rates_tensor = neutral_rates_tensor
        self.aa_parents_idxss = aa_parents_idxss
        self.aa_children_idxss = aa_children_idxss
        self.codon_mutation_indicators = codon_mutation_indicators
        self.masks = masks
        self.model_known_token_count = model_known_token_count
        self.sparse_neutral_rates = sparse_neutral_rates

        # Determine storage format
        self.use_sparse = sparse_neutral_rates is not None

        # Validate tensor shapes and consistency
        assert codon_parents_idxss.shape == codon_children_idxss.shape
        assert codon_parents_idxss.shape == codon_mutation_indicators.shape
        assert codon_parents_idxss.shape == masks.shape
        assert aa_parents_idxss.shape == aa_children_idxss.shape

        if self.use_sparse:
            # Validate sparse format
            assert "indices" in sparse_neutral_rates
            assert "values" in sparse_neutral_rates
            assert "n_mutations" in sparse_neutral_rates
            indices = sparse_neutral_rates["indices"]
            values = sparse_neutral_rates["values"]
            n_mutations = sparse_neutral_rates["n_mutations"]
            assert indices.shape[:2] == codon_parents_idxss.shape  # (N, L_codon, ...)
            assert values.shape[:2] == codon_parents_idxss.shape
            assert n_mutations.shape == codon_parents_idxss.shape
            assert (
                indices.shape[2] == values.shape[2]
            )  # max_mutations dimension matches
            assert indices.shape[3] == 2  # [parent_codon_idx, child_codon_idx]
        else:
            # Dense format validation
            assert neutral_rates_tensor.shape[:2] == codon_parents_idxss.shape
            assert neutral_rates_tensor.shape[2:] == (65, 65)

    def __len__(self):
        return len(self.nt_parents)

    def __getitem__(self, idx):
        """Return batch tensors for whichmut loss computation."""
        if self.use_sparse:
            # Return sparse format
            sparse_data = {
                "indices": self.sparse_neutral_rates["indices"][idx],
                "values": self.sparse_neutral_rates["values"][idx],
                "n_mutations": self.sparse_neutral_rates["n_mutations"][idx],
            }
            return (
                self.codon_parents_idxss[idx],
                self.codon_children_idxss[idx],
                sparse_data,  # Sparse neutral rates format
                self.aa_parents_idxss[idx],
                self.aa_children_idxss[idx],
                self.codon_mutation_indicators[idx],
                self.masks[idx],
            )
        else:
            # Return dense format (backward compatibility)
            return (
                self.codon_parents_idxss[idx],
                self.codon_children_idxss[idx],
                self.neutral_rates_tensor[idx],
                self.aa_parents_idxss[idx],
                self.aa_children_idxss[idx],
                self.codon_mutation_indicators[idx],
                self.masks[idx],
            )

    @classmethod
    def of_pcp_df(
        cls,
        pcp_df: pd.DataFrame,
        neutral_model_outputs: Dict[str, Any],  # Results from sub_rates_of_seq, etc.
        model_known_token_count: int,
    ):
        """Create WhichmutCodonDataset from PCP DataFrame and precomputed neutral model
        outputs.

        Args:
            pcp_df: DataFrame with 'nt_parent' and 'nt_child' columns
            neutral_model_outputs: Dict containing:
                - 'neutral_rates': precomputed λ_{j,c->c'} tensors
            model_known_token_count: Number of known tokens in selection model
        """
        # Extract sequences
        nt_parents = pcp_df["nt_parent"].reset_index(drop=True)
        nt_children = pcp_df["nt_child"].reset_index(drop=True)

        # Convert to codon indices and identify mutations
        codon_parents_idxss, codon_children_idxss, codon_mutation_indicators = (
            encode_codon_mutations(nt_parents, nt_children)
        )

        # Convert to amino acid indices
        aa_parents = translate_sequences(nt_parents)
        aa_children = translate_sequences(nt_children)

        # Convert AA sequences to index tensors
        max_aa_len = max(len(seq) for seq in aa_parents)
        n_sequences = len(aa_parents)

        aa_parents_idxss = torch.full(
            (n_sequences, max_aa_len), 20, dtype=torch.long
        )  # 20 is padding token
        aa_children_idxss = torch.full((n_sequences, max_aa_len), 20, dtype=torch.long)

        for i, (aa_parent, aa_child) in enumerate(zip(aa_parents, aa_children)):
            aa_parent_idxs = aa_idx_tensor_of_str_ambig(aa_parent)
            aa_child_idxs = aa_idx_tensor_of_str_ambig(aa_child)
            aa_parents_idxss[i, : len(aa_parent_idxs)] = aa_parent_idxs
            aa_children_idxss[i, : len(aa_child_idxs)] = aa_child_idxs

        # Create masks for valid positions
        masks = create_codon_masks(nt_parents, nt_children)

        # Extract precomputed neutral rates (can be dense tensor or sparse dict)
        neutral_rates_data = neutral_model_outputs["neutral_rates"]

        if isinstance(neutral_rates_data, dict):
            # Sparse format
            return cls(
                nt_parents,
                nt_children,
                codon_parents_idxss,
                codon_children_idxss,
                None,  # No dense tensor
                aa_parents_idxss,
                aa_children_idxss,
                codon_mutation_indicators,
                masks,
                model_known_token_count,
                sparse_neutral_rates=neutral_rates_data,
            )
        else:
            # Dense format (backward compatibility)
            return cls(
                nt_parents,
                nt_children,
                codon_parents_idxss,
                codon_children_idxss,
                neutral_rates_data,
                aa_parents_idxss,
                aa_children_idxss,
                codon_mutation_indicators,
                masks,
                model_known_token_count,
                sparse_neutral_rates=None,
            )


def compute_neutral_rates_for_sequences(
    nt_sequences: pd.Series,
    neutral_model_fn,  # e.g., sub_rates_of_seq from flu/scv2 modules
    **neutral_model_kwargs,
) -> torch.Tensor:
    """Compute neutral codon mutation rates λ_{j,c->c'} for sequences.

    Uses existing neutral model infrastructure to compute substitution rates,
    then converts to neutral rates at the codon level.

    Returns:
        Tensor of shape (N, L_codon, 65, 65) with neutral rates
    """
    neutral_rates_list = []

    for seq in tqdm(nt_sequences, desc="Computing neutral rates"):
        # Get nucleotide substitution rates from neutral model
        nt_rates = neutral_model_fn(seq, **neutral_model_kwargs)

        # Convert to codon-level neutral rates
        # For each codon position, compute neutral rate for each possible mutation
        L_codon = len(seq) // 3
        codon_neutral_rates = torch.zeros(
            L_codon, 65, 65
        )  # This is CPU tensor, will be moved to device later

        for codon_pos in range(L_codon):
            nt_start = codon_pos * 3
            parent_codon = seq[nt_start : nt_start + 3]

            # Skip if codon contains N or is not valid
            if "N" in parent_codon or parent_codon not in CODONS:
                continue

            parent_codon_idx = CODONS.index(parent_codon)

            # Use precomputed single mutation mapping
            for child_codon_idx, nt_pos, new_base in CODON_SINGLE_MUTATIONS[
                parent_codon_idx
            ]:
                # Get substitution rate for this specific nucleotide change
                global_nt_pos = nt_start + nt_pos
                if global_nt_pos < len(nt_rates):
                    rate = nt_rates[global_nt_pos, BASES_AND_N_TO_INDEX[new_base]]
                    codon_neutral_rates[
                        codon_pos, parent_codon_idx, child_codon_idx
                    ] = rate

        neutral_rates_list.append(codon_neutral_rates)

    return torch.stack(neutral_rates_list)
