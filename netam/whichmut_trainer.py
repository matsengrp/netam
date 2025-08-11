"""Whichmut trainer and loss function implementation.

This module implements the core whichmut loss function and trainer class that models
codon-level mutations using the formulation from viral-dasm-tex.

The mathematical model: p_{j,m}(X) = λ_{j,m}(X) * f_{j,a->a'}(X) / Z_j

Key components:
- WhichmutCodonDataset: Dataset class for codon-level data with precomputed neutral rates
- WhichmutTrainer: Trainer following existing framework patterns
- compute_whichmut_loss_batch(): Core loss computation
- compute_normalization_constants(): Efficient Z_j calculation
"""

import torch
import pandas as pd
from typing import Dict, Any, Optional
from tqdm import tqdm

from netam.codon_table import (
    FUNCTIONAL_CODON_SINGLE_MUTATIONS,
    CODON_SINGLE_MUTATIONS,
    encode_codon_mutations,
    create_codon_masks,
)
from netam.sequences import (
    translate_sequences,
    aa_idx_tensor_of_str_ambig,
    CODONS,
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
    return torch.tensor(0.0, device=values.device)


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


def compute_whichmut_loss_batch(
    selection_factors: torch.Tensor,
    neutral_rates_data: torch.Tensor,  # Can be dense tensor or sparse dict
    codon_parents_idxss: torch.Tensor,
    codon_children_idxss: torch.Tensor,
    codon_mutation_indicators: torch.Tensor,
    aa_parents_idxss: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Compute whichmut loss for a batch of sequences.

    Computes loss using the formula:
    Loss = -sum_k log(p_{j_k, c_k->c'_k}) over all observed mutations k
    where p_{j,c->c'} = λ_{j,c->c'} * f_{j,a->a'} / Z_j

    Args:
        selection_factors: (N, L_aa, 20) - model output selection factors
        neutral_rates_data: Either:
            - Dense tensor: (N, L_codon, 65, 65) - precomputed λ_{j,c->c'}
            - Sparse dict: {'indices': tensor, 'values': tensor, 'n_mutations': tensor}
        codon_parents_idxss: (N, L_codon) - parent codon indices
        codon_children_idxss: (N, L_codon) - child codon indices
        codon_mutation_indicators: (N, L_codon) - which sites mutated
        aa_parents_idxss: (N, L_aa) - parent AA indices
        masks: (N, L_codon) - valid positions
    Returns:
        Loss scalar for the batch
    """
    # Detect sparse vs dense format
    use_sparse = isinstance(neutral_rates_data, dict)

    N, L_codon = codon_parents_idxss.shape
    _, L_aa, _ = selection_factors.shape

    # Convert selection factors from log space to linear space
    # Selection model outputs log(f_{j,a->a'})
    linear_selection_factors = torch.exp(selection_factors)  # (N, L_aa, 20)

    # 1. Compute normalization constants Z_n for each sequence
    normalization_constants = compute_normalization_constants(
        linear_selection_factors, neutral_rates_data, codon_parents_idxss
    )  # (N,)

    # 2. For each observed mutation, compute log probability
    # For now, keep explicit loops for clarity and correctness verification
    log_probs = []
    mutations_count = 0

    for seq_idx in range(N):

        for codon_pos in range(L_codon):
            # Only process positions that actually mutated and are valid
            if (
                codon_mutation_indicators[seq_idx, codon_pos]
                and masks[seq_idx, codon_pos]
            ):
                mutations_count += 1
                parent_codon_idx = codon_parents_idxss[seq_idx, codon_pos]
                child_codon_idx = codon_children_idxss[seq_idx, codon_pos]

                # Get λ_{j,c->c'} for this specific mutation
                if use_sparse:
                    # Look up rate in sparse format
                    neutral_rate = get_sparse_neutral_rate(
                        neutral_rates_data,
                        seq_idx,
                        codon_pos,
                        parent_codon_idx,
                        child_codon_idx,
                    )
                else:
                    # Dense format lookup
                    neutral_rate = neutral_rates_data[
                        seq_idx, codon_pos, parent_codon_idx, child_codon_idx
                    ]

                # Get selection factor f_{j,a->a'} for the corresponding AA mutation
                # Map from codon position to AA position (assuming 1:1 mapping)
                aa_pos = codon_pos  # Assuming 1:1 mapping for now
                if aa_pos < L_aa:  # Ensure we don't go out of bounds
                    # Get child amino acid index from codon
                    from netam.codon_table import AA_IDX_FROM_CODON_IDX

                    child_aa_idx = AA_IDX_FROM_CODON_IDX[child_codon_idx.item()]

                    # Get selection factor for this AA change
                    selection_factor = linear_selection_factors[
                        seq_idx, aa_pos, child_aa_idx
                    ]

                    # Compute probability p_{j,c->c'} = λ * f / Z_n
                    Z = normalization_constants[seq_idx]
                    prob = (neutral_rate * selection_factor) / Z

                    # Add log probability to loss
                    log_probs.append(
                        torch.log(prob + 1e-10)
                    )  # Small epsilon for numerical stability

    if len(log_probs) == 0:
        # No mutations observed, return zero loss
        return torch.tensor(0.0, device=selection_factors.device, requires_grad=True)

    # Return negative log likelihood (we want to maximize likelihood = minimize negative log likelihood)
    total_log_likelihood = torch.stack(log_probs).sum()

    return -total_log_likelihood


def compute_normalization_constants(
    selection_factors: torch.Tensor,  # (N, L_aa, 20) - linear space f_{j,a->a'}
    neutral_rates_data: torch.Tensor,  # (N, L_codon, 65, 65) - λ_{j,c->c'} or sparse dict
    # aa_parents_idxss: torch.Tensor,  # (N, L_aa) - parent AA indices (unused for now)
    codon_parents_idxss: torch.Tensor,  # (N, L_codon) - parent codon indices
) -> torch.Tensor:
    """Compute normalization constants Z_j = sum_{m'} λ_{j,m'}(X) * f_{j,aa(m')}(X) for
    each codon position j.

    These constants normalize the whichmut probabilities to sum to 1 across all possible
    single-nucleotide mutations.

    Supports both dense and sparse neutral rates formats for optimal performance.
    """
    # Detect format
    use_sparse = isinstance(neutral_rates_data, dict)

    if use_sparse:
        # Extract batch dimensions from sparse data
        N, L_codon = neutral_rates_data["indices"].shape[:2]
        return compute_normalization_constants_sparse(
            selection_factors, neutral_rates_data, codon_parents_idxss
        )
    else:
        # Dense format
        N, L_codon, _, _ = neutral_rates_data.shape
        return compute_normalization_constants_dense(
            selection_factors, neutral_rates_data, codon_parents_idxss
        )


def compute_normalization_constants_dense(
    selection_factors: torch.Tensor,
    neutral_rates_tensor: torch.Tensor,
    codon_parents_idxss: torch.Tensor,
) -> torch.Tensor:
    """Dense implementation (original explicit loop version)."""
    from netam.codon_table import AA_IDX_FROM_CODON_IDX

    N, L_codon, _, _ = neutral_rates_tensor.shape

    # Initialize normalization constants
    Z = torch.zeros(N, device=selection_factors.device)

    # For each sequence, compute Z_n (normalization constant for the entire sequence)
    for seq_idx in range(N):
        Z_n = 0.0

        for codon_pos in range(L_codon):
            parent_codon_idx = codon_parents_idxss[seq_idx, codon_pos].item()

            for alt_codon_idx, _, _ in FUNCTIONAL_CODON_SINGLE_MUTATIONS[
                parent_codon_idx
            ]:
                # Get λ_{j,parent->possible child codon-mutation}
                neutral_rate = neutral_rates_tensor[
                    seq_idx, codon_pos, parent_codon_idx, alt_codon_idx
                ]

                # Skip zero rates in dense implementation (sparse already excludes them)
                if neutral_rate <= 0:
                    continue
                # Get corresponding amino acid for child codon
                child_aa_idx = AA_IDX_FROM_CODON_IDX[alt_codon_idx]

                # Skip stop codons and invalid amino acids (same as sparse implementation)
                if child_aa_idx >= 20:
                    continue

                # Get selection factor f_{j,aa(child)}
                aa_pos = codon_pos  # Assuming 1:1 mapping
                if aa_pos < selection_factors.shape[1]:
                    selection_factor = selection_factors[seq_idx, aa_pos, child_aa_idx]
                    Z_n += neutral_rate * selection_factor

        Z[seq_idx] = Z_n

    return Z


def compute_normalization_constants_sparse(
    selection_factors: torch.Tensor,
    sparse_neutral_rates: Dict[str, torch.Tensor],
    codon_parents_idxss: torch.Tensor,
) -> torch.Tensor:
    """Sparse implementation using vectorized operations for optimal performance.

    This is significantly more efficient than the dense version for large sequence
    lengths, reducing memory usage from O(N*L*65^2) to O(N*L*9) and enabling vectorized
    computation.
    """
    from netam.codon_table import AA_IDX_FROM_CODON_IDX

    # Extract sparse data components
    indices = sparse_neutral_rates["indices"]  # (N, L_codon, max_mutations, 2)
    values = sparse_neutral_rates["values"]  # (N, L_codon, max_mutations)
    n_mutations = sparse_neutral_rates["n_mutations"]  # (N, L_codon)

    N, L_codon, max_mutations, _ = indices.shape

    # Initialize normalization constants
    Z = torch.zeros(N, device=selection_factors.device)

    # Vectorized computation over all mutations
    # For each (seq, codon_pos, mutation_idx), compute neutral_rate * selection_factor

    # Create child AA indices from codon indices in indices tensor
    # indices[:, :, :, 1] contains child codon indices
    child_codon_indices = indices[:, :, :, 1]  # (N, L_codon, max_mutations)

    # Convert to child AA indices using vectorized lookup
    # Create a lookup tensor for efficient codon->AA mapping (65 codons total)
    codon_to_aa = torch.zeros(65, dtype=torch.long, device=child_codon_indices.device)
    for codon_idx, aa_idx in AA_IDX_FROM_CODON_IDX.items():
        codon_to_aa[codon_idx] = aa_idx

    # Use advanced indexing to map all codon indices to AA indices at once
    # Clamp to valid range first to prevent out-of-bounds access
    clamped_codon_indices = torch.clamp(child_codon_indices, 0, 64)
    child_aa_indices = codon_to_aa[clamped_codon_indices]

    # Get selection factors for all positions and child AAs
    # selection_factors is (N, L_aa, 20)
    # We need to gather the appropriate selection factors

    # Create position indices for gathering (assuming 1:1 codon->AA mapping)
    seq_indices = (
        torch.arange(N, device=selection_factors.device)
        .view(N, 1, 1)
        .expand(N, L_codon, max_mutations)
    )
    pos_indices = (
        torch.arange(L_codon, device=selection_factors.device)
        .view(1, L_codon, 1)
        .expand(N, L_codon, max_mutations)
    )

    # Clamp child_aa_indices to valid range to prevent index errors
    child_aa_indices_clamped = torch.clamp(child_aa_indices, 0, 19)

    # Gather selection factors: (N, L_codon, max_mutations)
    selection_factor_values = selection_factors[
        seq_indices, pos_indices, child_aa_indices_clamped
    ]

    # Compute products: neutral_rates * selection_factors
    # Both values and selection_factor_values are (N, L_codon, max_mutations)
    products = values * selection_factor_values  # (N, L_codon, max_mutations)

    # Create mask for valid mutations (only sum over actual mutations, not padding)
    mutation_mask = torch.arange(max_mutations, device=n_mutations.device).view(
        1, 1, max_mutations
    ) < n_mutations.unsqueeze(-1)

    # Also mask out stop codons and invalid AA indices (where child_aa_indices >= 20)
    valid_aa_mask = child_aa_indices < 20
    combined_mask = mutation_mask & valid_aa_mask

    # Apply mask and sum over mutations for each (seq, codon_pos)
    masked_products = products * combined_mask.float()  # (N, L_codon, max_mutations)
    codon_contributions = masked_products.sum(dim=-1)  # (N, L_codon)

    # Sum over all codon positions for each sequence to get Z_n
    Z = codon_contributions.sum(dim=-1)  # (N,)

    return Z


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
        codon_neutral_rates = torch.zeros(L_codon, 65, 65)

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
                from netam.sequences import BASES_AND_N_TO_INDEX

                if global_nt_pos < len(nt_rates):
                    rate = nt_rates[global_nt_pos, BASES_AND_N_TO_INDEX[new_base]]
                    codon_neutral_rates[
                        codon_pos, parent_codon_idx, child_codon_idx
                    ] = rate

        neutral_rates_list.append(codon_neutral_rates)

    return torch.stack(neutral_rates_list)


class WhichmutTrainer:
    """Trainer using whichmut loss following framework.py patterns."""

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def train_epoch(self, dataloader, **kwargs):
        """Train for one epoch."""
        return self._run_epoch(dataloader, training=True, **kwargs)

    def evaluate(self, dataloader, **kwargs):
        """Evaluate on validation data."""
        return self._run_epoch(dataloader, training=False, **kwargs)

    def _run_epoch(self, dataloader, training=False):
        """Core training/evaluation loop following framework.py:process_data_loader
        pattern."""
        total_loss = None
        total_samples = 0

        if training:
            self.model.train()
        else:
            self.model.eval()

        for batch_idx, batch_data in enumerate(
            tqdm(dataloader, desc="Processing batches")
        ):
            (
                codon_parents_idxss,
                codon_children_idxss,
                neutral_rates_data,  # Can be tensor or dict
                aa_parents_idxss,
                _,  # aa_children_idxss (unused)
                codon_mutation_indicators,
                masks,
            ) = batch_data

            batch_size = codon_parents_idxss.shape[0]

            # Warn about potentially slow batch sizes (only for dense format)
            if isinstance(neutral_rates_data, torch.Tensor):  # Dense format
                if batch_size > 10:
                    print(
                        f"⚠️  WARNING: Large batch size ({batch_size}) detected with DENSE format!"
                    )
                    print(
                        f"   This may be very slow due to inefficient dense implementation."
                    )
                    memory_mb = neutral_rates_data.numel() * 4 / 1024**2
                    print(
                        f"   Current batch uses ~{memory_mb:.1f} MB for neutral rates tensor."
                    )
                elif batch_size > 2:
                    print(
                        f"ℹ️  INFO: Processing batch {batch_idx + 1} with {batch_size} sequences (DENSE format)..."
                    )
            else:  # Sparse format
                if batch_size > 2:
                    indices = neutral_rates_data["indices"]
                    values = neutral_rates_data["values"]
                    n_mutations = neutral_rates_data["n_mutations"]
                    memory_mb = (
                        indices.numel() * 8
                        + values.numel() * 4
                        + n_mutations.numel() * 8
                    ) / 1024**2
                    print(
                        f"ℹ️  INFO: Processing batch {batch_idx + 1} with {batch_size} sequences (SPARSE format, ~{memory_mb:.1f} MB)..."
                    )

            # Selection model inference
            with torch.set_grad_enabled(training):
                if training:
                    # Training mode: gradient calculation with retry logic (framework.py:714-740)
                    max_grad_retries = 3
                    grad_retry_count = 0

                    while grad_retry_count < max_grad_retries:
                        try:
                            if self.optimizer:
                                self.optimizer.zero_grad()

                            selection_factors = self.model(aa_parents_idxss, masks)

                            # Compute whichmut loss using precomputed neutral rates
                            loss = compute_whichmut_loss_batch(
                                selection_factors,
                                neutral_rates_data,
                                codon_parents_idxss,
                                codon_children_idxss,
                                codon_mutation_indicators,
                                aa_parents_idxss,
                                masks,
                            )

                            if self.optimizer:
                                loss.backward()
                                # Check for invalid gradients (framework.py:731-736)
                                valid_gradients = all(
                                    (
                                        torch.isfinite(p.grad).all()
                                        if p.grad is not None
                                        else True
                                    )
                                    for p in self.model.parameters()
                                )

                                if valid_gradients:
                                    self.optimizer.step()
                                    break
                                else:
                                    grad_retry_count += 1
                                    if grad_retry_count < max_grad_retries:
                                        print(
                                            f"Retrying gradient calculation ({grad_retry_count}/{max_grad_retries}) with loss {loss.item()}"
                                        )
                                        # Recompute loss exactly as in framework.py:722
                                        selection_factors = self.model(
                                            aa_parents_idxss, masks
                                        )
                                        loss = compute_whichmut_loss_batch(
                                            selection_factors,
                                            neutral_rates_data,
                                            codon_parents_idxss,
                                            codon_children_idxss,
                                            codon_mutation_indicators,
                                            aa_parents_idxss,
                                            masks,
                                        )
                                    else:
                                        raise ValueError(
                                            "Exceeded maximum gradient retries!"
                                        )
                        except Exception as e:
                            grad_retry_count += 1
                            if grad_retry_count >= max_grad_retries:
                                raise e
                            print(
                                f"Error during gradient calculation, retrying ({grad_retry_count}/{max_grad_retries}): {e}"
                            )

                else:
                    # Evaluation mode: no gradients
                    selection_factors = self.model(aa_parents_idxss, masks)
                    loss = compute_whichmut_loss_batch(
                        selection_factors,
                        neutral_rates_data,
                        codon_parents_idxss,
                        codon_children_idxss,
                        codon_mutation_indicators,
                        aa_parents_idxss,
                        masks,
                    )

                # Accumulate loss (framework.py:730-735 pattern)
                batch_samples = aa_parents_idxss.shape[0]
                if total_loss is None:
                    total_loss = loss * batch_samples
                else:
                    total_loss += loss * batch_samples
                total_samples += batch_samples

        # Return average loss
        if total_samples == 0:
            return torch.tensor(0.0)
        average_loss = total_loss / total_samples
        return average_loss


def create_whichmut_trainer(
    model_config_yaml: str,  # Path to YAML configuration file
    device: torch.device,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    optimizer_name: Optional[str] = None,
    train_mode: bool = True,
):
    """Factory that handles whichmut trainer setup complexity.

    Following existing patterns for dynamic optimizer instantiation and parameter
    override system.
    """
    import yaml
    import torch.optim as optim
    from netam.model_factory import create_selection_model_from_dict

    # Load configuration from YAML
    with open(model_config_yaml, "r") as f:
        config = yaml.safe_load(f)

    # Parameter override system
    final_lr = (
        learning_rate
        if learning_rate is not None
        else config.get("learning_rate", 0.001)
    )
    final_weight_decay = (
        weight_decay if weight_decay is not None else config.get("weight_decay", 0.0)
    )
    final_optimizer_name = (
        optimizer_name
        if optimizer_name is not None
        else config.get("optimizer_name", "Adam")
    )

    # Load and create model from config
    model = create_selection_model_from_dict(config, device)

    # Create optimizer if in training mode using dynamic instantiation
    optimizer = None
    if train_mode and final_optimizer_name:
        if hasattr(optim, final_optimizer_name):
            optimizer_class = getattr(optim, final_optimizer_name)
            optimizer = optimizer_class(
                model.parameters(), lr=final_lr, weight_decay=final_weight_decay
            )
        else:
            available = [name for name in dir(optim) if name[0].isupper()]
            raise ValueError(
                f"Unknown optimizer: {final_optimizer_name}. Available: {available}"
            )

    return WhichmutTrainer(model, optimizer)
