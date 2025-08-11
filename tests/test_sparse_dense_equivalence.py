#!/usr/bin/env python
"""Focused unit tests to verify sparse and dense WhichMut operations are numerically
equivalent."""

import pytest
import torch

from netam.whichmut_trainer import (
    compute_normalization_constants_dense,
    compute_normalization_constants_sparse,
    get_sparse_neutral_rate,
)
from netam.codon_table import FUNCTIONAL_CODON_SINGLE_MUTATIONS, AA_IDX_FROM_CODON_IDX
from netam.sequences import CODONS


def create_equivalent_data(
    batch_size: int = 2,
    sequence_length: int = 5,
    device: torch.device = torch.device("cpu"),
):
    """Create dense and sparse data that should be exactly equivalent."""

    # Use AAA codon for simplicity (has 8 functional mutations)
    aaa_idx = CODONS.index("AAA")

    # Create codon data
    codon_parents_idxss = torch.full(
        (batch_size, sequence_length), aaa_idx, dtype=torch.long, device=device
    )

    # Create selection factors (20 amino acids)
    torch.manual_seed(42)  # For reproducibility
    selection_factors = (
        torch.randn(batch_size, sequence_length, 20, device=device) * 0.1
    )
    linear_selection_factors = torch.exp(selection_factors)

    # Create dense neutral rates
    dense_rates = torch.zeros(batch_size, sequence_length, 65, 65, device=device)

    # Create sparse data structures
    functional_mutations = FUNCTIONAL_CODON_SINGLE_MUTATIONS[aaa_idx]
    n_mutations = len(functional_mutations)

    indices = torch.full(
        (batch_size, sequence_length, n_mutations, 2),
        -1,
        dtype=torch.long,
        device=device,
    )
    values = torch.zeros(batch_size, sequence_length, n_mutations, device=device)
    n_mutations_tensor = torch.full(
        (batch_size, sequence_length), n_mutations, dtype=torch.long, device=device
    )

    # Fill both dense and sparse with identical data
    for seq_idx in range(batch_size):
        for pos in range(sequence_length):
            for mut_idx, (child_idx, _, _) in enumerate(functional_mutations):
                # Only include mutations to valid amino acids (0-19)
                child_aa_idx = AA_IDX_FROM_CODON_IDX[child_idx]
                if child_aa_idx >= 20:  # Skip stop codons
                    continue

                # Use deterministic rate based on position and mutation
                rate = 0.01 * (1 + 0.1 * seq_idx + 0.05 * pos + 0.02 * mut_idx)

                # Dense format
                dense_rates[seq_idx, pos, aaa_idx, child_idx] = rate

                # Sparse format
                indices[seq_idx, pos, mut_idx, 0] = aaa_idx
                indices[seq_idx, pos, mut_idx, 1] = child_idx
                values[seq_idx, pos, mut_idx] = rate

    # Update n_mutations to reflect only valid mutations (non-stop codons)
    actual_n_mutations = 0
    for child_idx, _, _ in functional_mutations:
        if AA_IDX_FROM_CODON_IDX[child_idx] < 20:
            actual_n_mutations += 1

    n_mutations_tensor.fill_(actual_n_mutations)

    sparse_rates = {
        "indices": indices,
        "values": values,
        "n_mutations": n_mutations_tensor,
    }

    return {
        "linear_selection_factors": linear_selection_factors,
        "dense_rates": dense_rates,
        "sparse_rates": sparse_rates,
        "codon_parents_idxss": codon_parents_idxss,
    }


class TestSparseDenseEquivalence:
    """Test exact numerical equivalence between sparse and dense implementations."""

    def test_simple_normalization_equivalence(self):
        """Test normalization constants with simple equivalent data."""
        data = create_equivalent_data(batch_size=2, sequence_length=3)

        dense_Z = compute_normalization_constants_dense(
            data["linear_selection_factors"],
            data["dense_rates"],
            data["codon_parents_idxss"],
        )

        sparse_Z = compute_normalization_constants_sparse(
            data["linear_selection_factors"],
            data["sparse_rates"],
            data["codon_parents_idxss"],
        )

        print(f"Dense Z: {dense_Z}")
        print(f"Sparse Z: {sparse_Z}")
        print(f"Difference: {dense_Z - sparse_Z}")
        print(f"Relative difference: {(dense_Z - sparse_Z) / dense_Z}")

        assert torch.allclose(
            dense_Z, sparse_Z, rtol=1e-6, atol=1e-8
        ), f"Normalization mismatch: Dense={dense_Z}, Sparse={sparse_Z}"

    def test_sparse_lookup_correctness(self):
        """Test that sparse lookup returns correct values."""
        data = create_equivalent_data(batch_size=1, sequence_length=1)

        aaa_idx = CODONS.index("AAA")
        functional_mutations = FUNCTIONAL_CODON_SINGLE_MUTATIONS[aaa_idx]

        # Test lookup for first valid mutation
        for child_idx, _, _ in functional_mutations:
            if AA_IDX_FROM_CODON_IDX[child_idx] < 20:  # Valid AA
                # Get expected value from dense
                expected_rate = data["dense_rates"][0, 0, aaa_idx, child_idx]

                # Get value from sparse lookup
                sparse_rate = get_sparse_neutral_rate(
                    data["sparse_rates"], 0, 0, aaa_idx, child_idx
                )

                if expected_rate > 0:  # Only test non-zero rates
                    assert torch.allclose(
                        sparse_rate, expected_rate, rtol=1e-6
                    ), f"Lookup mismatch for codon {child_idx}: sparse={sparse_rate}, dense={expected_rate}"
                break

    def test_different_batch_sizes(self):
        """Test equivalence across different batch sizes."""
        for batch_size in [1, 4, 8]:
            data = create_equivalent_data(batch_size=batch_size, sequence_length=3)

            dense_Z = compute_normalization_constants_dense(
                data["linear_selection_factors"],
                data["dense_rates"],
                data["codon_parents_idxss"],
            )

            sparse_Z = compute_normalization_constants_sparse(
                data["linear_selection_factors"],
                data["sparse_rates"],
                data["codon_parents_idxss"],
            )

            assert torch.allclose(
                dense_Z, sparse_Z, rtol=1e-6, atol=1e-8
            ), f"Batch size {batch_size} mismatch: Dense={dense_Z}, Sparse={sparse_Z}"

    def test_zero_rates_handling(self):
        """Test that zero rates are handled consistently."""
        data = create_equivalent_data(batch_size=2, sequence_length=3)

        # Set some rates to zero in both formats
        aaa_idx = CODONS.index("AAA")
        functional_mutations = FUNCTIONAL_CODON_SINGLE_MUTATIONS[aaa_idx]
        first_child_idx = functional_mutations[0][0]

        # Zero out first mutation in dense format
        data["dense_rates"][0, 0, aaa_idx, first_child_idx] = 0.0

        # Zero out first mutation in sparse format and adjust count
        data["sparse_rates"]["values"][0, 0, 0] = 0.0

        # Test that they still match
        dense_Z = compute_normalization_constants_dense(
            data["linear_selection_factors"],
            data["dense_rates"],
            data["codon_parents_idxss"],
        )

        sparse_Z = compute_normalization_constants_sparse(
            data["linear_selection_factors"],
            data["sparse_rates"],
            data["codon_parents_idxss"],
        )

        assert torch.allclose(
            dense_Z, sparse_Z, rtol=1e-6, atol=1e-8
        ), f"Zero rate handling mismatch: Dense={dense_Z}, Sparse={sparse_Z}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
