"""Whichmut trainer and loss function implementation.

This module implements the core whichmut loss function and trainer class that models
codon-level mutations using the formulation from viral-dasm-tex.

The mathematical model: p_{j,m}(X) = λ_{j,m}(X) * f_{j,a->a'}(X) / Z_j

Key components:
- WhichmutTrainer: Trainer following existing framework patterns
- compute_whichmut_loss_batch(): Core loss computation
- compute_normalization_constants(): Efficient Z_j calculation
"""

import torch
from typing import Dict, Optional
from tqdm import tqdm

from netam.codon_table import FUNCTIONAL_CODON_SINGLE_MUTATIONS
from netam.whichmut_dataset import (
    SparseWhichmutCodonDataset,
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
            - Dense tensor: (N, L_codon, 65) - precomputed λ rates per child codon
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

    # If using sparse format, ensure data is on the correct device
    if use_sparse:
        device = selection_factors.device
        neutral_rates_data = {
            "indices": neutral_rates_data["indices"].to(device),
            "values": neutral_rates_data["values"].to(device),
            "n_possible_mutations": neutral_rates_data["n_possible_mutations"].to(
                device
            ),
        }

    # 1. Compute normalization constants Z_n for each sequence
    normalization_constants = compute_normalization_constants(
        linear_selection_factors, neutral_rates_data, codon_parents_idxss
    )  # (N,)

    # 2. Vectorized computation of log probabilities for all mutations
    # Create boolean mask for valid mutations (mutated AND masked positions)
    mutation_mask = codon_mutation_indicators & masks  # (N, L_codon)

    # Early exit if no mutations
    if not mutation_mask.any():
        return torch.tensor(0.0, device=selection_factors.device, requires_grad=True)

    # Get indices of all mutations using nonzero()
    mutation_indices = mutation_mask.nonzero(
        as_tuple=False
    )  # (num_mutations, 2) -> [seq_idx, codon_pos]

    if mutation_indices.numel() == 0:
        return torch.tensor(0.0, device=selection_factors.device, requires_grad=True)

    # Extract sequence and position indices for all mutations
    seq_indices = mutation_indices[:, 0]  # (num_mutations,)
    pos_indices = mutation_indices[:, 1]  # (num_mutations,)

    # Gather parent and child codon indices for all mutations
    parent_codon_indices = codon_parents_idxss[
        seq_indices, pos_indices
    ]  # (num_mutations,)
    child_codon_indices = codon_children_idxss[
        seq_indices, pos_indices
    ]  # (num_mutations,)

    # Get neutral rates for all mutations
    if use_sparse:
        # Vectorized sparse lookup
        neutral_rates = SparseWhichmutCodonDataset.get_neutral_rates_vectorized(
            neutral_rates_data,
            seq_indices,
            pos_indices,
            parent_codon_indices,
            child_codon_indices,
        )
    else:
        # Dense format: look up rate TO the child codon
        # First verify child is reachable from parent via single mutation
        # For efficiency in the vectorized case, we just look up the rate
        # (it will be 0 if not reachable)
        neutral_rates = neutral_rates_data[
            seq_indices, pos_indices, child_codon_indices
        ]  # (num_mutations,)

    # Convert child codon indices to AA indices
    from netam.codon_table import AA_IDX_FROM_CODON_IDX

    # Create vectorized codon->AA lookup tensor
    device = child_codon_indices.device
    codon_to_aa = torch.zeros(65, dtype=torch.long, device=device)
    for codon_idx, aa_idx in AA_IDX_FROM_CODON_IDX.items():
        codon_to_aa[codon_idx] = aa_idx

    # Convert all child codon indices to AA indices
    child_aa_indices = codon_to_aa[child_codon_indices]  # (num_mutations,)

    # Filter out mutations to stop codons (AA index >= 20)
    valid_aa_mask = child_aa_indices < 20

    if not valid_aa_mask.any():
        return torch.tensor(0.0, device=selection_factors.device, requires_grad=True)

    # Apply filter to keep only valid AA mutations
    seq_indices = seq_indices[valid_aa_mask]
    pos_indices = pos_indices[valid_aa_mask]
    neutral_rates = neutral_rates[valid_aa_mask]
    child_aa_indices = child_aa_indices[valid_aa_mask]

    # Ensure AA positions are within bounds
    aa_pos_mask = pos_indices < L_aa
    if not aa_pos_mask.any():
        return torch.tensor(0.0, device=selection_factors.device, requires_grad=True)

    # Apply AA position filter
    seq_indices = seq_indices[aa_pos_mask]
    pos_indices = pos_indices[aa_pos_mask]
    neutral_rates = neutral_rates[aa_pos_mask]
    child_aa_indices = child_aa_indices[aa_pos_mask]

    # Gather selection factors for all valid mutations
    selection_factors_mut = linear_selection_factors[
        seq_indices, pos_indices, child_aa_indices
    ]  # (num_valid_mutations,)

    # Gather normalization constants for all mutations
    Z_values = normalization_constants[seq_indices]  # (num_valid_mutations,)

    # Compute probabilities: p = (λ * f) / Z
    probs = (neutral_rates * selection_factors_mut) / Z_values

    # Compute log probabilities with numerical stability
    log_probs = torch.log(probs + 1e-10)

    # Return negative log likelihood
    total_log_likelihood = log_probs.sum()
    return -total_log_likelihood


def compute_whichmut_loss_batch_iterative(
    selection_factors: torch.Tensor,
    neutral_rates_data: torch.Tensor,
    codon_parents_idxss: torch.Tensor,
    codon_children_idxss: torch.Tensor,
    codon_mutation_indicators: torch.Tensor,
    aa_parents_idxss: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Original iterative implementation for comparison/testing."""
    use_sparse = isinstance(neutral_rates_data, dict)
    N, L_codon = codon_parents_idxss.shape
    _, L_aa, _ = selection_factors.shape

    linear_selection_factors = torch.exp(selection_factors)

    if use_sparse:
        device = selection_factors.device
        neutral_rates_data = {
            "indices": neutral_rates_data["indices"].to(device),
            "values": neutral_rates_data["values"].to(device),
            "n_possible_mutations": neutral_rates_data["n_possible_mutations"].to(
                device
            ),
        }

    normalization_constants = compute_normalization_constants(
        linear_selection_factors, neutral_rates_data, codon_parents_idxss
    )

    # Original nested loop implementation
    log_probs = []
    for seq_idx in range(N):
        for codon_pos in range(L_codon):
            if (
                codon_mutation_indicators[seq_idx, codon_pos]
                and masks[seq_idx, codon_pos]
            ):
                parent_codon_idx = codon_parents_idxss[seq_idx, codon_pos]
                child_codon_idx = codon_children_idxss[seq_idx, codon_pos]

                if use_sparse:
                    neutral_rate = SparseWhichmutCodonDataset.get_neutral_rate(
                        neutral_rates_data,
                        seq_idx,
                        codon_pos,
                        parent_codon_idx,
                        child_codon_idx,
                    )
                else:
                    # Dense format: look up rate TO the child codon
                    neutral_rate = neutral_rates_data[
                        seq_idx, codon_pos, child_codon_idx
                    ]

                aa_pos = codon_pos
                if aa_pos < L_aa:
                    from netam.codon_table import AA_IDX_FROM_CODON_IDX

                    child_aa_idx = AA_IDX_FROM_CODON_IDX[child_codon_idx.item()]
                    selection_factor = linear_selection_factors[
                        seq_idx, aa_pos, child_aa_idx
                    ]
                    Z = normalization_constants[seq_idx]
                    prob = (neutral_rate * selection_factor) / Z
                    log_probs.append(torch.log(prob + 1e-10))

    if len(log_probs) == 0:
        return torch.tensor(0.0, device=selection_factors.device, requires_grad=True)

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
        # Dense format - now (N, L_codon, 65) instead of (N, L_codon, 65, 65)
        N, L_codon, _ = neutral_rates_data.shape
        return compute_normalization_constants_dense(
            selection_factors, neutral_rates_data, codon_parents_idxss
        )


def compute_normalization_constants_dense(
    selection_factors: torch.Tensor,
    neutral_rates_tensor: torch.Tensor,
    codon_parents_idxss: torch.Tensor,
) -> torch.Tensor:
    """Dense implementation with optimized storage (N, L, 65)."""
    from netam.codon_table import AA_IDX_FROM_CODON_IDX

    N, L_codon, _ = neutral_rates_tensor.shape  # Now shape is (N, L_codon, 65)

    # Initialize normalization constants
    Z = torch.zeros(N, device=selection_factors.device)

    # For each sequence, compute Z_n (normalization constant for the entire sequence)
    for seq_idx in range(N):
        Z_n = 0.0

        for codon_pos in range(L_codon):
            parent_codon_idx = codon_parents_idxss[seq_idx, codon_pos].item()

            # Iterate over possible single mutations from this parent codon
            for alt_codon_idx, _, _ in FUNCTIONAL_CODON_SINGLE_MUTATIONS[
                parent_codon_idx
            ]:
                # Get λ rate TO this child codon
                neutral_rate = neutral_rates_tensor[seq_idx, codon_pos, alt_codon_idx]

                # Skip zero rates (child codon not reachable from parent)
                if neutral_rate <= 0:
                    continue

                # Get corresponding amino acid for child codon
                child_aa_idx = AA_IDX_FROM_CODON_IDX[alt_codon_idx]

                # Skip stop codons and invalid amino acids
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

    # Extract sparse data components - they should already be on the correct device
    indices = sparse_neutral_rates["indices"]  # (N, L_codon, max_mutations, 2)
    values = sparse_neutral_rates["values"]  # (N, L_codon, max_mutations)
    n_possible_mutations = sparse_neutral_rates["n_possible_mutations"]  # (N, L_codon)

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
    mutation_mask = torch.arange(
        max_mutations, device=n_possible_mutations.device
    ).view(1, 1, max_mutations) < n_possible_mutations.unsqueeze(-1)

    # Also mask out stop codons and invalid AA indices (where child_aa_indices >= 20)
    valid_aa_mask = child_aa_indices < 20
    combined_mask = mutation_mask & valid_aa_mask

    # Apply mask and sum over mutations for each (seq, codon_pos)
    masked_products = products * combined_mask.float()  # (N, L_codon, max_mutations)
    codon_contributions = masked_products.sum(dim=-1)  # (N, L_codon)

    # Sum over all codon positions for each sequence to get Z_n
    Z = codon_contributions.sum(dim=-1)  # (N,)

    return Z


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

    def _move_batch_to_device(self, batch_data, device):
        """Move all batch data to the specified device.

        Handles both dense and sparse neutral rates formats, ensuring all
        tensors are properly moved to the target device.

        Args:
            batch_data: Tuple of batch tensors from dataloader
            device: Target device (cuda/cpu)

        Returns:
            Tuple of moved tensors in same order as input
        """
        (
            codon_parents_idxss,
            codon_children_idxss,
            neutral_rates_data,
            aa_parents_idxss,
            aa_children_idxss,
            codon_mutation_indicators,
            masks,
        ) = batch_data

        # Move standard tensors
        codon_parents_idxss = codon_parents_idxss.to(device)
        codon_children_idxss = codon_children_idxss.to(device)
        aa_parents_idxss = aa_parents_idxss.to(device)
        aa_children_idxss = aa_children_idxss.to(device)
        codon_mutation_indicators = codon_mutation_indicators.to(device)
        masks = masks.to(device)

        # Move neutral rates data (handles both dense and sparse formats)
        if isinstance(neutral_rates_data, torch.Tensor):
            # Dense format
            neutral_rates_data = neutral_rates_data.to(device)
        else:
            # Sparse format - move all components
            neutral_rates_data = {
                "indices": neutral_rates_data["indices"].to(device),
                "values": neutral_rates_data["values"].to(device),
                "n_possible_mutations": neutral_rates_data["n_possible_mutations"].to(
                    device
                ),
            }

        return (
            codon_parents_idxss,
            codon_children_idxss,
            neutral_rates_data,
            aa_parents_idxss,
            aa_children_idxss,
            codon_mutation_indicators,
            masks,
        )

    def _log_batch_info(self, batch_idx, batch_size, neutral_rates_data):
        """Log information about batch processing for debugging.

        Provides warnings for potentially slow configurations and
        memory usage information.

        Args:
            batch_idx: Current batch index
            batch_size: Number of sequences in batch
            neutral_rates_data: Neutral rates (tensor or dict) to analyze
        """
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
                n_possible_mutations = neutral_rates_data["n_possible_mutations"]
                memory_mb = (
                    indices.numel() * 8
                    + values.numel() * 4
                    + n_possible_mutations.numel() * 8
                ) / 1024**2
                print(
                    f"ℹ️  INFO: Processing batch {batch_idx + 1} with {batch_size} sequences (SPARSE format, ~{memory_mb:.1f} MB)..."
                )

    def _compute_loss(self, batch_data):
        """Compute loss for a batch of data.

        Args:
            batch_data: Tuple of (codon_parents, codon_children, neutral_rates,
                                  aa_parents, aa_children, mutation_indicators, masks)

        Returns:
            Loss tensor
        """
        (
            codon_parents_idxss,
            codon_children_idxss,
            neutral_rates_data,
            aa_parents_idxss,
            _,  # aa_children_idxss (unused)
            codon_mutation_indicators,
            masks,
        ) = batch_data

        # Forward pass through model
        selection_factors = self.model(aa_parents_idxss, masks)

        # Compute whichmut loss
        loss = compute_whichmut_loss_batch(
            selection_factors,
            neutral_rates_data,
            codon_parents_idxss,
            codon_children_idxss,
            codon_mutation_indicators,
            aa_parents_idxss,
            masks,
        )

        return loss

    def _train_step_with_retry(self, batch_data, max_retries=3):
        """Execute training step with gradient retry logic.

        Attempts to compute gradients and update model parameters,
        retrying if invalid gradients are detected.

        Args:
            batch_data: Batch data tuple (already on correct device)
            max_retries: Maximum number of gradient computation retries

        Returns:
            Computed loss tensor

        Raises:
            ValueError: If maximum retries exceeded
        """
        grad_retry_count = 0

        while grad_retry_count < max_retries:
            try:
                if self.optimizer:
                    self.optimizer.zero_grad()

                # Compute loss
                loss = self._compute_loss(batch_data)

                if self.optimizer:
                    loss.backward()

                    # Check for invalid gradients
                    valid_gradients = self._check_gradients()

                    if valid_gradients:
                        self.optimizer.step()
                        return loss
                    else:
                        grad_retry_count += 1
                        if grad_retry_count < max_retries:
                            print(
                                f"Retrying gradient calculation ({grad_retry_count}/{max_retries}) "
                                f"with loss {loss.item()}"
                            )
                        else:
                            raise ValueError("Exceeded maximum gradient retries!")
                else:
                    # No optimizer, just return loss
                    return loss

            except Exception as e:
                grad_retry_count += 1
                if grad_retry_count >= max_retries:
                    raise e
                print(
                    f"Error during gradient calculation, retrying "
                    f"({grad_retry_count}/{max_retries}): {e}"
                )

        raise ValueError("Failed to compute gradients after all retries")

    def _check_gradients(self):
        """Check if all model gradients are valid (finite).

        Returns:
            True if all gradients are valid, False otherwise
        """
        return all(
            (torch.isfinite(p.grad).all() if p.grad is not None else True)
            for p in self.model.parameters()
        )

    def _run_epoch(self, dataloader, training=False):
        """Core training/evaluation loop.

        Processes all batches in the dataloader, computing loss and
        optionally updating model parameters.

        Args:
            dataloader: PyTorch DataLoader with batch data
            training: Whether to perform gradient updates

        Returns:
            Average loss across all samples
        """
        total_loss = None
        total_samples = 0

        # Get device from model
        device = next(self.model.parameters()).device

        # Set model mode
        if training:
            self.model.train()
        else:
            self.model.eval()

        # Process each batch
        for batch_idx, batch_data in enumerate(
            tqdm(dataloader, desc="Processing batches")
        ):
            # Move batch to device
            batch_data = self._move_batch_to_device(batch_data, device)

            # Extract batch size for logging
            batch_size = batch_data[0].shape[0]  # codon_parents_idxss

            # Log batch information
            self._log_batch_info(
                batch_idx, batch_size, batch_data[2]
            )  # neutral_rates_data

            # Compute loss (with gradient updates if training)
            with torch.set_grad_enabled(training):
                if training:
                    loss = self._train_step_with_retry(batch_data)
                else:
                    loss = self._compute_loss(batch_data)

            # Accumulate loss for averaging
            if total_loss is None:
                total_loss = loss * batch_size
            else:
                total_loss += loss * batch_size
            total_samples += batch_size

        # Return average loss
        if total_samples == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / total_samples


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
