"""Dataset class for whichmut training with precomputed neutral rates.

This module provides the WhichmutCodonDataset class for storing and managing codon-level
parent/child sequences with precomputed neutral rates for codon-level selection
modeling.

## Sparse Data Structure Format

The sparse neutral rates format is designed for memory efficiency when dealing with
large-scale sequence datasets. Instead of storing full 65x65 transition matrices
(most entries are zero), it stores only the non-zero single-nucleotide mutation rates.

### Core Data Structure

The sparse format uses a dictionary with three tensors:

```python
sparse_neutral_rates = {
    'indices': torch.LongTensor,     # Shape: (N, L, max_mutations, 2)
    'values': torch.Tensor,          # Shape: (N, L, max_mutations)
    'n_possible_mutations': torch.LongTensor  # Shape: (N, L)
}
```

### Components Explained

1. **`indices` tensor (N, L, max_mutations, 2)**:
   - Stores [parent_codon_idx, child_codon_idx] pairs for each possible mutation
   - Uses full codon indices (0-64 range), not compressed indices
   - `max_mutations` is always 9 (3 positions × 3 alternative bases per position)
   - Only the first `n_possible_mutations[seq, pos]` entries are valid per position

2. **`values` tensor (N, L, max_mutations)**:
   - Contains the actual neutral rates λ_{j,c->c'} for each mutation
   - Aligned with the `indices` tensor - same indexing scheme
   - Only the first `n_possible_mutations[seq, pos]` entries are valid per position

3. **`n_possible_mutations` tensor (N, L)**:
   - Indicates how many valid single-nucleotide mutations are stored per position
   - Typically 7-9 mutations per codon (after filtering stop codons)
   - The name emphasizes these are *possible* mutations, not observed mutations

### FUNCTIONAL_CODON_SINGLE_MUTATIONS Lookup Table

The sparse data structure is built using `FUNCTIONAL_CODON_SINGLE_MUTATIONS` from
`netam.codon_table`, which has the structure:

```python
FUNCTIONAL_CODON_SINGLE_MUTATIONS = {
    parent_codon_idx: [
        (child_codon_idx, nt_position, new_base),
        ...
    ]
}
```

**Key properties**:
- Excludes mutations to stop codons (only functional codons included)
- Each parent codon maps to 7-9 possible single-nucleotide mutations
- `nt_position` is 0, 1, or 2 (position within the codon)
- `new_base` is the replacement nucleotide

### Example Sparse Data

For a codon AAA (index 0) at one sequence position:

```python
# AAA has 8 functional single mutations (excluding stop codons)
indices[0, 0, :8, :] = [
    [0, 1],   # AAA -> AAC (Lys -> Asn)
    [0, 2],   # AAA -> AAG (Lys -> Lys, synonymous)
    [0, 4],   # AAA -> ACA (Lys -> Thr)
    [0, 8],   # AAA -> AGA (Lys -> Arg)
    [0, 16],  # AAA -> CAA (Lys -> Gln)
    [0, 32],  # AAA -> GAA (Lys -> Glu)
    [0, 48],  # AAA -> TAA would be stop - excluded
    [0, 56],  # AAA -> TCA (Lys -> Ser)
    # ... (8th mutation)
]

values[0, 0, :8] = [0.01, 0.015, 0.012, ...]  # Corresponding neutral rates
n_possible_mutations[0, 0] = 8  # 8 valid mutations stored
```

### Usage in Loss Computation

The sparse format integrates seamlessly with whichmut loss computation:

1. **Normalization constants**: Sum over all possible mutations per position
2. **Loss calculation**: Efficient lookup of observed mutation rates
3. **Vectorization**: All mutations processed in parallel using advanced indexing

See `netam.whichmut_trainer` for implementation details.
"""

import torch
import pandas as pd
from typing import Dict
from abc import ABC, abstractmethod
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


class WhichmutCodonDataset(ABC):
    """Abstract base class for whichmut training datasets.

    Provides the common interface for codon-level parent/child sequences with
    precomputed neutral rates. Designed for codon-level selection modeling.

    Does not enforce single mutation per PCP - handles arbitrary numbers of mutations.
    """

    def __init__(
        self,
        nt_parents: pd.Series,
        nt_children: pd.Series,
        codon_parents_idxss: torch.Tensor,
        codon_children_idxss: torch.Tensor,
        aa_parents_idxss: torch.Tensor,
        aa_children_idxss: torch.Tensor,
        codon_mutation_indicators: torch.Tensor,
        masks: torch.Tensor,
        model_known_token_count: int,
    ):
        """Initialize common dataset attributes.

        Args:
            nt_parents: Parent nucleotide sequences
            nt_children: Child nucleotide sequences
            codon_parents_idxss: Parent codons as indices (N, L_codon)
            codon_children_idxss: Child codons as indices (N, L_codon)
            aa_parents_idxss: Parent AAs as indices (N, L_aa)
            aa_children_idxss: Child AAs as indices (N, L_aa)
            codon_mutation_indicators: Which codon sites mutated (N, L_codon)
            masks: Valid codon positions (N, L_codon)
            model_known_token_count: Number of known tokens in selection model
        """
        self.nt_parents = nt_parents
        self.nt_children = nt_children
        self.codon_parents_idxss = codon_parents_idxss
        self.codon_children_idxss = codon_children_idxss
        self.aa_parents_idxss = aa_parents_idxss
        self.aa_children_idxss = aa_children_idxss
        self.codon_mutation_indicators = codon_mutation_indicators
        self.masks = masks
        self.model_known_token_count = model_known_token_count

        # Validate common tensor shapes
        assert codon_parents_idxss.shape == codon_children_idxss.shape
        assert codon_parents_idxss.shape == codon_mutation_indicators.shape
        assert codon_parents_idxss.shape == masks.shape
        assert aa_parents_idxss.shape == aa_children_idxss.shape

    def __len__(self):
        return len(self.nt_parents)

    @abstractmethod
    def __getitem__(self, idx):
        """Return batch tensors for whichmut loss computation.

        Must be implemented by subclasses to return format-specific data.
        """
        pass

    @staticmethod
    def _prepare_common_data(pcp_df: pd.DataFrame):
        """Extract and prepare common sequence data from PCP DataFrame.

        Helper method used by concrete classes to avoid code duplication.
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

        aa_parents_idxss = torch.full((n_sequences, max_aa_len), 20, dtype=torch.long)
        aa_children_idxss = torch.full((n_sequences, max_aa_len), 20, dtype=torch.long)

        for i, (aa_parent, aa_child) in enumerate(zip(aa_parents, aa_children)):
            aa_parent_idxs = aa_idx_tensor_of_str_ambig(aa_parent)
            aa_child_idxs = aa_idx_tensor_of_str_ambig(aa_child)
            aa_parents_idxss[i, : len(aa_parent_idxs)] = aa_parent_idxs
            aa_children_idxss[i, : len(aa_child_idxs)] = aa_child_idxs

        # Create masks for valid positions
        masks = create_codon_masks(nt_parents, nt_children)

        return (
            nt_parents,
            nt_children,
            codon_parents_idxss,
            codon_children_idxss,
            aa_parents_idxss,
            aa_children_idxss,
            codon_mutation_indicators,
            masks,
        )


class DenseWhichmutCodonDataset(WhichmutCodonDataset):
    """Dense implementation of WhichmutCodonDataset.

    Stores neutral rates in a dense tensor of shape (N, L_codon, 65). More memory
    intensive but simpler data structure.
    """

    def __init__(
        self,
        nt_parents: pd.Series,
        nt_children: pd.Series,
        codon_parents_idxss: torch.Tensor,
        codon_children_idxss: torch.Tensor,
        aa_parents_idxss: torch.Tensor,
        aa_children_idxss: torch.Tensor,
        codon_mutation_indicators: torch.Tensor,
        masks: torch.Tensor,
        model_known_token_count: int,
        neutral_rates_tensor: torch.Tensor,
    ):
        """Initialize dense dataset with neutral rates tensor.

        Args:
            neutral_rates_tensor: Dense tensor (N, L_codon, 65) with rates
                                  for each possible child codon
        """
        super().__init__(
            nt_parents,
            nt_children,
            codon_parents_idxss,
            codon_children_idxss,
            aa_parents_idxss,
            aa_children_idxss,
            codon_mutation_indicators,
            masks,
            model_known_token_count,
        )

        self.neutral_rates_tensor = neutral_rates_tensor

        # Validate dense format
        assert neutral_rates_tensor.shape[:2] == codon_parents_idxss.shape
        assert neutral_rates_tensor.shape[2] == 65  # One rate per child codon

    def __getitem__(self, idx):
        """Return batch tensors with dense neutral rates."""
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
        dense_neutral_rates: torch.Tensor,
        model_known_token_count: int,
    ):
        """Create DenseWhichmutCodonDataset from PCP DataFrame and dense neutral rates.

        Args:
            pcp_df: DataFrame with 'nt_parent' and 'nt_child' columns
            dense_neutral_rates: Dense tensor (N, L_codon, 65) with precomputed λ rates
            model_known_token_count: Number of known tokens in selection model

        Returns:
            DenseWhichmutCodonDataset instance
        """
        # Prepare common sequence data
        (
            nt_parents,
            nt_children,
            codon_parents_idxss,
            codon_children_idxss,
            aa_parents_idxss,
            aa_children_idxss,
            codon_mutation_indicators,
            masks,
        ) = cls._prepare_common_data(pcp_df)

        return cls(
            nt_parents,
            nt_children,
            codon_parents_idxss,
            codon_children_idxss,
            aa_parents_idxss,
            aa_children_idxss,
            codon_mutation_indicators,
            masks,
            model_known_token_count,
            neutral_rates_tensor=dense_neutral_rates,
        )

    @classmethod
    def compute_neutral_rates_from_sequences(
        cls,
        nt_sequences: pd.Series,
        neutral_model_fn,  # e.g., sub_rates_of_seq from flu/scv2 modules
        **neutral_model_kwargs,
    ) -> torch.Tensor:
        """Compute dense format neutral codon mutation rates λ_{j,c->c'} for sequences.

        Uses existing neutral model infrastructure to compute substitution rates,
        then converts to neutral rates at the codon level in dense format.

        Args:
            nt_sequences: Series of nucleotide sequences
            neutral_model_fn: Function to compute nucleotide substitution rates
            **neutral_model_kwargs: Additional arguments for neutral_model_fn

        Returns:
            Tensor of shape (N, L_codon, 65) with neutral rates for each possible child codon.
            For each position, stores the rate to mutate TO each child codon from the
            actual parent codon at that position.
        """
        neutral_rates_list = []

        for seq in tqdm(nt_sequences, desc="Computing dense neutral rates"):
            # Get nucleotide substitution rates from neutral model
            nt_rates = neutral_model_fn(seq, **neutral_model_kwargs)

            # Convert to codon-level neutral rates
            # For each codon position, compute neutral rate for each possible mutation
            L_codon = len(seq) // 3
            codon_neutral_rates = torch.zeros(
                L_codon, 65
            )  # One rate per possible child codon

            for codon_pos in range(L_codon):
                nt_start = codon_pos * 3
                parent_codon = seq[nt_start : nt_start + 3]

                # Skip if codon contains N or is not valid
                if "N" in parent_codon or parent_codon not in CODONS:
                    continue

                parent_codon_idx = CODONS.index(parent_codon)

                # Use precomputed single mutation mapping
                # Store the rate TO each reachable child codon
                for child_codon_idx, nt_pos, new_base in CODON_SINGLE_MUTATIONS[
                    parent_codon_idx
                ]:
                    # Get substitution rate for this specific nucleotide change
                    global_nt_pos = nt_start + nt_pos
                    if global_nt_pos < len(nt_rates):
                        rate = nt_rates[global_nt_pos, BASES_AND_N_TO_INDEX[new_base]]
                        # Store rate at child codon index (not parent x child)
                        codon_neutral_rates[codon_pos, child_codon_idx] = rate

            neutral_rates_list.append(codon_neutral_rates)

        return torch.stack(neutral_rates_list)


class SparseWhichmutCodonDataset(WhichmutCodonDataset):
    """Sparse implementation of WhichmutCodonDataset.

    Stores only non-zero neutral rates for memory efficiency. Optimized for large-scale
    datasets with many sequences.
    """

    def __init__(
        self,
        nt_parents: pd.Series,
        nt_children: pd.Series,
        codon_parents_idxss: torch.Tensor,
        codon_children_idxss: torch.Tensor,
        aa_parents_idxss: torch.Tensor,
        aa_children_idxss: torch.Tensor,
        codon_mutation_indicators: torch.Tensor,
        masks: torch.Tensor,
        model_known_token_count: int,
        sparse_neutral_rates: Dict[str, torch.Tensor],
    ):
        """Initialize sparse dataset with structured neutral rates.

        Args:
            sparse_neutral_rates: Dict with 'indices', 'values', 'n_possible_mutations'
                                 tensors storing only non-zero rates
        """
        super().__init__(
            nt_parents,
            nt_children,
            codon_parents_idxss,
            codon_children_idxss,
            aa_parents_idxss,
            aa_children_idxss,
            codon_mutation_indicators,
            masks,
            model_known_token_count,
        )

        self.sparse_neutral_rates = sparse_neutral_rates

        # Validate sparse format
        assert "indices" in sparse_neutral_rates
        assert "values" in sparse_neutral_rates
        assert "n_possible_mutations" in sparse_neutral_rates

        indices = sparse_neutral_rates["indices"]
        values = sparse_neutral_rates["values"]
        n_possible_mutations = sparse_neutral_rates["n_possible_mutations"]

        assert indices.shape[:2] == codon_parents_idxss.shape
        assert values.shape[:2] == codon_parents_idxss.shape
        assert n_possible_mutations.shape == codon_parents_idxss.shape
        assert indices.shape[2] == values.shape[2]  # max_mutations dimension
        assert indices.shape[3] == 2  # [parent_codon_idx, child_codon_idx]

    def __getitem__(self, idx):
        """Return batch tensors with sparse neutral rates."""
        sparse_data = {
            "indices": self.sparse_neutral_rates["indices"][idx],
            "values": self.sparse_neutral_rates["values"][idx],
            "n_possible_mutations": self.sparse_neutral_rates["n_possible_mutations"][
                idx
            ],
        }
        return (
            self.codon_parents_idxss[idx],
            self.codon_children_idxss[idx],
            sparse_data,
            self.aa_parents_idxss[idx],
            self.aa_children_idxss[idx],
            self.codon_mutation_indicators[idx],
            self.masks[idx],
        )

    @classmethod
    def of_pcp_df(
        cls,
        pcp_df: pd.DataFrame,
        sparse_neutral_rates: Dict[str, torch.Tensor],
        model_known_token_count: int,
    ):
        """Create SparseWhichmutCodonDataset from PCP DataFrame and sparse neutral
        rates.

        Args:
            pcp_df: DataFrame with 'nt_parent' and 'nt_child' columns
            sparse_neutral_rates: Dict with 'indices', 'values', 'n_possible_mutations'
                                  tensors storing only non-zero rates
            model_known_token_count: Number of known tokens in selection model

        Returns:
            SparseWhichmutCodonDataset instance
        """
        # Prepare common sequence data
        (
            nt_parents,
            nt_children,
            codon_parents_idxss,
            codon_children_idxss,
            aa_parents_idxss,
            aa_children_idxss,
            codon_mutation_indicators,
            masks,
        ) = cls._prepare_common_data(pcp_df)

        return cls(
            nt_parents,
            nt_children,
            codon_parents_idxss,
            codon_children_idxss,
            aa_parents_idxss,
            aa_children_idxss,
            codon_mutation_indicators,
            masks,
            model_known_token_count,
            sparse_neutral_rates=sparse_neutral_rates,
        )

    @classmethod
    def compute_neutral_rates_from_sequences(
        cls,
        nt_sequences: pd.Series,
        neutral_model_fn,  # e.g., sub_rates_of_seq from flu/scv2 modules
        **neutral_model_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute sparse format neutral codon mutation rates λ_{j,c->c'} for sequences.

        Uses existing neutral model infrastructure to compute substitution rates,
        then converts to neutral rates at the codon level in sparse format.

        Args:
            nt_sequences: Series of nucleotide sequences
            neutral_model_fn: Function to compute nucleotide substitution rates
            **neutral_model_kwargs: Additional arguments for neutral_model_fn

        Returns:
            Dictionary with:
                - 'indices': (M, 4) tensor with [seq_idx, pos, parent_idx, child_idx]
                - 'values': (M,) tensor with non-zero neutral rates
                - 'n_possible_mutations': (N,) tensor with mutation counts per sequence
        """
        all_indices = []
        all_values = []
        n_possible_mutations_list = []

        for seq_idx, seq in enumerate(
            tqdm(nt_sequences, desc="Computing sparse neutral rates")
        ):
            # Get nucleotide substitution rates from neutral model
            nt_rates = neutral_model_fn(seq, **neutral_model_kwargs)

            # Track indices and values for this sequence
            seq_indices = []
            seq_values = []
            seq_mutation_count = 0

            # Convert to codon-level neutral rates
            L_codon = len(seq) // 3

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
                        if rate > 0:  # Only store non-zero rates
                            seq_indices.append(
                                [seq_idx, codon_pos, parent_codon_idx, child_codon_idx]
                            )
                            seq_values.append(rate)
                            seq_mutation_count += 1

            # Add this sequence's data to the overall lists
            if seq_indices:  # Only if there are non-zero rates
                all_indices.extend(seq_indices)
                all_values.extend(seq_values)
            n_possible_mutations_list.append(seq_mutation_count)

        # Convert to tensors
        if all_indices:
            indices_tensor = torch.tensor(all_indices, dtype=torch.long)
            values_tensor = torch.tensor(all_values, dtype=torch.float32)
        else:
            # Handle edge case of no non-zero rates
            indices_tensor = torch.zeros((0, 4), dtype=torch.long)
            values_tensor = torch.zeros((0,), dtype=torch.float32)

        n_possible_mutations_tensor = torch.tensor(
            n_possible_mutations_list, dtype=torch.long
        )

        return {
            "indices": indices_tensor,
            "values": values_tensor,
            "n_possible_mutations": n_possible_mutations_tensor,
        }

    @staticmethod
    def get_neutral_rate(
        sparse_data, seq_idx, codon_pos, parent_codon_idx, child_codon_idx
    ):
        """Look up neutral rate in sparse format.

        Args:
            sparse_data: Dict with 'indices', 'values', 'n_possible_mutations'
            seq_idx: Sequence index
            codon_pos: Codon position
            parent_codon_idx: Parent codon index
            child_codon_idx: Child codon index

        Returns:
            Neutral rate for the specified transition, or 0.0 if not found
        """
        indices = sparse_data["indices"][
            seq_idx, codon_pos
        ]  # Shape: (max_mutations, 2)
        values = sparse_data["values"][seq_idx, codon_pos]  # Shape: (max_mutations,)
        n_possible_mutations = sparse_data["n_possible_mutations"][
            seq_idx, codon_pos
        ].item()

        # Search for matching parent->child transition in the sparse data
        for i in range(n_possible_mutations):
            if indices[i, 0] == parent_codon_idx and indices[i, 1] == child_codon_idx:
                return values[i]

        # Not found - return 0 (this might indicate an error in sparse encoding)
        return torch.tensor(0.0, device=values.device, dtype=values.dtype)

    @staticmethod
    def get_neutral_rates_vectorized(
        sparse_data, seq_indices, pos_indices, parent_codon_indices, child_codon_indices
    ):
        """Vectorized lookup of neutral rates in sparse format.

        Args:
            sparse_data: Dict with 'indices', 'values', 'n_possible_mutations'
            seq_indices: (num_mutations,) - sequence indices
            pos_indices: (num_mutations,) - codon position indices
            parent_codon_indices: (num_mutations,) - parent codon indices
            child_codon_indices: (num_mutations,) - child codon indices

        Returns:
            (num_mutations,) tensor of neutral rates
        """
        num_mutations = seq_indices.shape[0]
        device = seq_indices.device

        # Initialize result tensor
        neutral_rates = torch.zeros(
            num_mutations, device=device, dtype=sparse_data["values"].dtype
        )

        # For each mutation, search in the sparse data
        for i in range(num_mutations):
            seq_idx = seq_indices[i]
            pos_idx = pos_indices[i]
            parent_idx = parent_codon_indices[i]
            child_idx = child_codon_indices[i]

            # Get sparse data for this (seq, pos) pair
            indices = sparse_data["indices"][seq_idx, pos_idx]  # (max_mutations, 2)
            values = sparse_data["values"][seq_idx, pos_idx]  # (max_mutations,)
            n_muts = sparse_data["n_possible_mutations"][seq_idx, pos_idx].item()

            # Search for matching transition
            for j in range(n_muts):
                if indices[j, 0] == parent_idx and indices[j, 1] == child_idx:
                    neutral_rates[i] = values[j]
                    break

        return neutral_rates
