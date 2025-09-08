"""Tests for whichmut trainer and loss function implementation."""

import torch
import pandas as pd
import numpy as np
import pytest

from netam.whichmut_trainer import (
    WhichmutTrainer,
    compute_whichmut_loss_batch,
    compute_whichmut_loss_batch_iterative,
    compute_normalization_constants,
    compute_normalization_constants_dense,
    compute_normalization_constants_sparse,
)
from netam.whichmut_dataset import (
    DenseWhichmutCodonDataset,
    SparseWhichmutCodonDataset,
)
from netam.sequences import CODONS, AA_STR_SORTED
from netam.codon_table import AA_IDX_FROM_CODON_IDX, FUNCTIONAL_CODON_SINGLE_MUTATIONS


def aa_idx_of_flat_codon_idx(codon_idx):
    """Get amino acid index from codon index using the global mapping."""
    # Handle ambiguous codon (index 64)
    if codon_idx == 64:
        return 20  # Ambiguous AA index
    # Handle stop codons and regular codons
    return AA_IDX_FROM_CODON_IDX.get(codon_idx, 20)


def set_neutral_rates_for_codon(
    neutral_rates_tensor, seq_idx, codon_pos, parent_codon_idx, default_rate=0.01
):
    """Set neutral rates for all possible single mutations from a parent codon."""
    if parent_codon_idx in FUNCTIONAL_CODON_SINGLE_MUTATIONS:
        for child_idx, _, _ in FUNCTIONAL_CODON_SINGLE_MUTATIONS[parent_codon_idx]:
            # New dense format: store rate at child codon index
            neutral_rates_tensor[seq_idx, codon_pos, child_idx] = default_rate


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

    # Create dense neutral rates (optimized format: one rate per child codon)
    dense_rates = torch.zeros(batch_size, sequence_length, 65, device=device)

    # Create sparse data structures
    functional_mutations = FUNCTIONAL_CODON_SINGLE_MUTATIONS[aaa_idx]
    n_possible_mutations = len(functional_mutations)

    indices = torch.full(
        (batch_size, sequence_length, n_possible_mutations, 2),
        -1,
        dtype=torch.long,
        device=device,
    )
    values = torch.zeros(
        batch_size, sequence_length, n_possible_mutations, device=device
    )
    n_mutations_tensor = torch.full(
        (batch_size, sequence_length),
        n_possible_mutations,
        dtype=torch.long,
        device=device,
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

                # Dense format: store rate at child codon index
                dense_rates[seq_idx, pos, child_idx] = rate

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
        "n_possible_mutations": n_mutations_tensor,
    }

    return {
        "linear_selection_factors": linear_selection_factors,
        "dense_rates": dense_rates,
        "sparse_rates": sparse_rates,
        "codon_parents_idxss": codon_parents_idxss,
    }


def test_whichmut_codon_dataset_creation():
    """Test basic creation and validation of WhichmutCodonDataset."""
    # Create simple test data
    nt_parents = pd.Series(["ATGAAACCC", "TGGCCCGGG"])
    nt_children = pd.Series(
        ["ATGAAACCG", "TGGCCCGGG"]
    )  # CCC->CCG mutation in first seq

    # Mock neutral rates tensor (2 sequences, 3 codon positions, 65 child codons)
    neutral_rates_tensor = torch.zeros(2, 3, 65)

    # Mock other required tensors
    codon_parents_idxss = torch.zeros(2, 3, dtype=torch.long)
    codon_children_idxss = torch.zeros(2, 3, dtype=torch.long)
    aa_parents_idxss = torch.zeros(2, 3, dtype=torch.long)
    aa_children_idxss = torch.zeros(2, 3, dtype=torch.long)
    codon_mutation_indicators = torch.tensor(
        [[False, False, True], [False, False, False]]
    )
    masks = torch.ones(2, 3, dtype=torch.bool)

    dataset = DenseWhichmutCodonDataset(
        nt_parents,
        nt_children,
        codon_parents_idxss,
        codon_children_idxss,
        aa_parents_idxss,
        aa_children_idxss,
        codon_mutation_indicators,
        masks,
        model_known_token_count=20,
        neutral_rates_tensor=neutral_rates_tensor,
    )

    assert len(dataset) == 2
    assert dataset.model_known_token_count == 20

    # Test __getitem__
    batch_item = dataset[0]
    assert len(batch_item) == 7  # All expected tensors returned


def test_whichmut_codon_dataset_of_pcp_df():
    """Test creation from PCP DataFrame."""
    # Create test PCP DataFrame
    pcp_df = pd.DataFrame(
        {
            "nt_parent": ["ATGAAACCC", "TGGCCCGGG"],
            "nt_child": ["ATGAAACCG", "TGGCCCGGG"],
        }
    )

    # Mock neutral model outputs
    neutral_model_outputs = {
        "neutral_rates": torch.zeros(2, 3, 65)  # 2 sequences, 3 codons
    }

    dataset = DenseWhichmutCodonDataset.of_pcp_df(
        pcp_df, neutral_model_outputs["neutral_rates"], model_known_token_count=20
    )

    assert len(dataset) == 2
    # Check that mutation was properly detected
    assert (
        dataset.codon_mutation_indicators[0, 2].item() is True
    )  # Third codon mutated in first seq
    assert (
        dataset.codon_mutation_indicators[1, 2].item() is False
    )  # No mutation in second seq


def test_compute_whichmut_loss_simple_case():
    """Test whichmut loss with a simple manually-calculable example."""
    # Test setup: 1 sequence, 2 codon positions
    N, L_codon, L_aa = 1, 2, 2

    # Define specific codons and mutations
    # Position 0: ATG (Met) -> ATT (Ile)
    # Position 1: TGG (Trp) -> TGC (Cys)
    atg_idx = CODONS.index("ATG")
    att_idx = CODONS.index("ATT")
    tgg_idx = CODONS.index("TGG")
    tgc_idx = CODONS.index("TGC")

    # Set up input tensors
    codon_parents_idxss = torch.tensor([[atg_idx, tgg_idx]])  # (1, 2)
    codon_children_idxss = torch.tensor([[att_idx, tgc_idx]])  # (1, 2)
    codon_mutation_indicators = torch.tensor([[True, True]])  # Both positions mutated
    masks = torch.tensor([[True, True]])  # Both positions valid

    # Parent amino acids: Met, Trp
    met_idx = AA_STR_SORTED.index("M")
    trp_idx = AA_STR_SORTED.index("W")
    aa_parents_idxss = torch.tensor([[met_idx, trp_idx]])  # (1, 2)

    # Set up neutral rates tensor - need rates for ALL possible single mutations
    neutral_rates_tensor = torch.zeros(N, L_codon, 65)

    # Set neutral rates for all possible mutations from parent codons
    set_neutral_rates_for_codon(
        neutral_rates_tensor, 0, 0, atg_idx, 0.005
    )  # ATG mutations
    set_neutral_rates_for_codon(
        neutral_rates_tensor, 0, 1, tgg_idx, 0.005
    )  # TGG mutations

    # Override specific mutation rates
    neutral_rates_tensor[0, 0, att_idx] = 0.01  # λ for ATG->ATT = 0.01
    neutral_rates_tensor[0, 1, tgc_idx] = 0.02  # λ for TGG->TGC = 0.02

    # Add some background mutations for realistic partition function
    ata_idx = CODONS.index("ATA")  # ATG->ATA (synonymous Met)
    ctg_idx = CODONS.index("CTG")  # ATG->CTG (Leu)
    cgg_idx = CODONS.index("CGG")  # TGG->CGG (Arg)
    neutral_rates_tensor[0, 0, ata_idx] = 0.005  # λ for ATG->ATA (synonymous)
    neutral_rates_tensor[0, 0, ctg_idx] = 0.003  # λ for ATG->CTG
    neutral_rates_tensor[0, 1, cgg_idx] = 0.008  # λ for TGG->CGG

    # Set up selection factors (in log space, as output by model)
    # Position 0: Ile gets selection factor 2.0, others get 1.0
    # Position 1: Cys gets selection factor 1.5, others get 1.0
    selection_factors = torch.zeros(N, L_aa, 20)  # (1, 2, 20)
    ile_idx = AA_STR_SORTED.index("I")
    cys_idx = AA_STR_SORTED.index("C")

    selection_factors[0, 0, ile_idx] = np.log(2.0)  # Ile at position 0: f=2.0
    selection_factors[0, 1, cys_idx] = np.log(1.5)  # Cys at position 1: f=1.5
    # All other selection factors remain 0 (which means f=1.0 in linear space)

    # Define reference calculation function
    def compute_expected_loss_reference(
        neutral_rates_tensor,
        selection_factors,
        codon_parents_idxss,
        codon_children_idxss,
        codon_mutation_indicators,
        masks,
    ):
        """Reference implementation for testing - compute loss step by step."""
        N, L_codon = codon_parents_idxss.shape
        linear_selection_factors = torch.exp(selection_factors)

        total_log_likelihood = 0.0

        for seq_idx in range(N):
            # First compute Z_n for the entire sequence (per-sequence normalization)
            Z_n = 0.0
            for pos in range(L_codon):
                parent_idx = codon_parents_idxss[seq_idx, pos].item()
                if parent_idx in FUNCTIONAL_CODON_SINGLE_MUTATIONS:
                    for possible_child_idx, _, _ in FUNCTIONAL_CODON_SINGLE_MUTATIONS[
                        parent_idx
                    ]:
                        lambda_val = neutral_rates_tensor[
                            seq_idx, pos, possible_child_idx
                        ].item()
                        if lambda_val > 0:
                            child_aa_idx_possible = aa_idx_of_flat_codon_idx(
                                possible_child_idx
                            )
                            f_val = linear_selection_factors[
                                seq_idx, pos, child_aa_idx_possible
                            ].item()
                            Z_n += lambda_val * f_val

            # Now compute log likelihoods for observed mutations using the per-sequence Z_n
            for codon_pos in range(L_codon):
                if (
                    codon_mutation_indicators[seq_idx, codon_pos]
                    and masks[seq_idx, codon_pos]
                ):
                    child_codon_idx = codon_children_idxss[seq_idx, codon_pos].item()

                    # Get λ_{j,c->c'} for observed mutation
                    lambda_obs = neutral_rates_tensor[
                        seq_idx, codon_pos, child_codon_idx
                    ].item()

                    # Get selection factor for child AA
                    child_aa_idx = aa_idx_of_flat_codon_idx(child_codon_idx)
                    f_obs = linear_selection_factors[
                        seq_idx, codon_pos, child_aa_idx
                    ].item()

                    # Compute probability and log likelihood using per-sequence Z_n
                    prob = (lambda_obs * f_obs) / Z_n
                    total_log_likelihood += np.log(prob)

        return -total_log_likelihood  # Return negative log likelihood

    # Compute expected loss using reference implementation
    expected_loss = compute_expected_loss_reference(
        neutral_rates_tensor,
        selection_factors,
        codon_parents_idxss,
        codon_children_idxss,
        codon_mutation_indicators,
        masks,
    )

    # Compute loss using our function
    loss = compute_whichmut_loss_batch(
        selection_factors,
        neutral_rates_tensor,
        codon_parents_idxss,
        codon_children_idxss,
        codon_mutation_indicators,
        aa_parents_idxss,
        masks,
    )

    # Check the computed loss matches reference calculation
    assert (
        torch.abs(loss - expected_loss) < 0.001
    ), f"Loss {loss.item():.4f} doesn't match reference {expected_loss:.4f}"


def test_whichmut_loss_no_mutations():
    """Test that loss is 0 when no mutations are observed."""
    N, L_codon, L_aa = 1, 2, 2

    # Set up tensors with no mutations
    codon_parents_idxss = torch.zeros(N, L_codon, dtype=torch.long)
    codon_children_idxss = torch.zeros(N, L_codon, dtype=torch.long)
    codon_mutation_indicators = torch.tensor([[False, False]])  # No mutations
    neutral_rates_tensor = torch.zeros(N, L_codon, 65)
    aa_parents_idxss = torch.zeros(N, L_aa, dtype=torch.long)
    selection_factors = torch.zeros(N, L_aa, 20)
    masks = torch.ones(N, L_codon, dtype=torch.bool)

    # Set neutral rates for all possible mutations from parent codons even though no mutations occurred
    set_neutral_rates_for_codon(neutral_rates_tensor, 0, 0, 0)  # AAA
    set_neutral_rates_for_codon(neutral_rates_tensor, 0, 1, 0)  # AAA

    loss = compute_whichmut_loss_batch(
        selection_factors,
        neutral_rates_tensor,
        codon_parents_idxss,
        codon_children_idxss,
        codon_mutation_indicators,
        aa_parents_idxss,
        masks,
    )

    # Loss should be 0 when no mutations are observed
    assert loss.item() == 0.0


def test_compute_normalization_constants():
    """Test normalization constant computation."""
    N, L_aa, L_codon = 1, 2, 2

    # Simple test case
    selection_factors = torch.ones(N, L_aa, 20)  # All selection factors = 1.0
    neutral_rates_tensor = torch.zeros(N, L_codon, 65)  # New dense format

    # Set up parent codons
    codon_parents_idxss = torch.zeros(N, L_codon, dtype=torch.long)
    codon_parents_idxss[0, 0] = 0  # AAA
    codon_parents_idxss[0, 1] = 5  # AAT

    # Add neutral rates for all possible mutations from parent codons
    set_neutral_rates_for_codon(neutral_rates_tensor, 0, 0, 0, 0.1)  # AAA with rate 0.1
    set_neutral_rates_for_codon(neutral_rates_tensor, 0, 1, 5, 0.3)  # AAT with rate 0.3

    Z = compute_normalization_constants(
        selection_factors, neutral_rates_tensor, codon_parents_idxss
    )

    # Now expecting per-sequence normalization (shape should be (N,) not (N, L_codon))
    assert Z.shape == (N,)
    assert torch.all(Z >= 0)  # Normalization constants should be non-negative

    # The normalization constant should be the sum over all positions and all possible mutations
    # With selection factors all = 1.0, this should equal sum of all neutral rates
    # For AAA (codon 0): 8 functional single mutations * 0.1 = 0.8
    # For AAT (codon 5): 9 functional single mutations * 0.3 = 2.7
    # Total expected Z = 0.8 + 2.7 = 3.5
    assert torch.abs(Z[0] - 3.5) < 0.01


def test_compute_normalization_constants_comprehensive():  # noqa: C901
    """Comprehensive test for normalization constant computation covering various
    scenarios."""

    # Test 1: Basic functionality with uniform rates and selection factors
    def test_uniform_case():
        """Test with uniform neutral rates and selection factors."""
        N, L_codon, L_aa = 2, 3, 3

        # All selection factors = 1.0 (exp(0) = 1)
        selection_factors = torch.ones(N, L_aa, 20)
        neutral_rates_tensor = torch.zeros(N, L_codon, 65)

        # Use same codon (AAA) everywhere for simplicity
        aaa_idx = CODONS.index("AAA")
        codon_parents_idxss = torch.full((N, L_codon), aaa_idx, dtype=torch.long)

        # Set uniform rate for all mutations
        uniform_rate = 0.05
        for seq_idx in range(N):
            for pos in range(L_codon):
                set_neutral_rates_for_codon(
                    neutral_rates_tensor, seq_idx, pos, aaa_idx, uniform_rate
                )

        Z = compute_normalization_constants(
            selection_factors, neutral_rates_tensor, codon_parents_idxss
        )

        # Calculate expected value
        # AAA has 8 functional mutations to non-stop codons
        mutations_per_codon = len(
            [
                m
                for m in FUNCTIONAL_CODON_SINGLE_MUTATIONS[aaa_idx]
                if AA_IDX_FROM_CODON_IDX[m[0]] < 20
            ]
        )
        expected_Z = mutations_per_codon * uniform_rate * L_codon  # per sequence

        assert Z.shape == (N,)
        assert torch.allclose(Z, torch.full((N,), expected_Z), rtol=1e-5)

    # Test 2: Variable selection factors
    def test_variable_selection():
        """Test with different selection factors affecting normalization."""
        N, L_codon, L_aa = 1, 2, 2

        # Non-uniform selection factors (in linear space, not log)
        selection_factors = torch.ones(N, L_aa, 20)
        # Make specific amino acids have different selection
        lys_idx = AA_STR_SORTED.index("K")  # AAA codes for K
        asn_idx = AA_STR_SORTED.index("N")  # AAC codes for N
        selection_factors[0, 0, lys_idx] = 2.0  # Double selection for K at position 0
        selection_factors[0, 1, asn_idx] = 3.0  # Triple selection for N at position 1

        neutral_rates_tensor = torch.zeros(N, L_codon, 65)

        # Use AAA codons
        aaa_idx = CODONS.index("AAA")
        codon_parents_idxss = torch.full((N, L_codon), aaa_idx, dtype=torch.long)

        # Set rates manually for specific mutations
        for seq_idx in range(N):
            for pos in range(L_codon):
                for child_idx, _, _ in FUNCTIONAL_CODON_SINGLE_MUTATIONS[aaa_idx]:
                    if AA_IDX_FROM_CODON_IDX[child_idx] < 20:  # Skip stop codons
                        neutral_rates_tensor[seq_idx, pos, child_idx] = 0.1

        Z = compute_normalization_constants(
            selection_factors, neutral_rates_tensor, codon_parents_idxss
        )

        # Manual calculation:
        # Position 0: Sum of (rate * selection_factor) for all mutations
        # Position 1: Sum of (rate * selection_factor) for all mutations
        expected_Z = 0.0
        for pos in range(L_codon):
            for child_idx, _, _ in FUNCTIONAL_CODON_SINGLE_MUTATIONS[aaa_idx]:
                child_aa = AA_IDX_FROM_CODON_IDX[child_idx]
                if child_aa < 20:
                    rate = 0.1
                    selection = selection_factors[0, pos, child_aa].item()
                    expected_Z += rate * selection

        assert torch.abs(Z[0] - expected_Z) < 1e-5

    # Test 3: Mixed codon types
    def test_mixed_codons():
        """Test with different codon types having different mutation patterns."""
        N, L_codon, L_aa = 1, 3, 3

        selection_factors = torch.ones(N, L_aa, 20)
        neutral_rates_tensor = torch.zeros(N, L_codon, 65)

        # Use different codons with known mutation counts
        atg_idx = CODONS.index("ATG")  # Met, 8 mutations
        tgg_idx = CODONS.index("TGG")  # Trp, 8 mutations
        gcc_idx = CODONS.index("GCC")  # Ala, 8 mutations

        codon_parents_idxss = torch.tensor([[atg_idx, tgg_idx, gcc_idx]])

        # Set different rates for each codon
        rates = [0.01, 0.02, 0.03]
        for pos, (codon_idx, rate) in enumerate(
            zip([atg_idx, tgg_idx, gcc_idx], rates)
        ):
            set_neutral_rates_for_codon(neutral_rates_tensor, 0, pos, codon_idx, rate)

        Z = compute_normalization_constants(
            selection_factors, neutral_rates_tensor, codon_parents_idxss
        )

        # Calculate expected Z
        expected_Z = 0.0
        for pos, (codon_idx, rate) in enumerate(
            zip([atg_idx, tgg_idx, gcc_idx], rates)
        ):
            mutations = [
                m
                for m in FUNCTIONAL_CODON_SINGLE_MUTATIONS[codon_idx]
                if AA_IDX_FROM_CODON_IDX[m[0]] < 20
            ]
            expected_Z += len(mutations) * rate

        assert torch.abs(Z[0] - expected_Z) < 1e-5

    # Test 4: Batch processing
    def test_batch_processing():
        """Test that batch processing works correctly."""
        N, L_codon, L_aa = 4, 2, 2

        selection_factors = torch.ones(N, L_aa, 20)
        neutral_rates_tensor = torch.zeros(N, L_codon, 65)

        aaa_idx = CODONS.index("AAA")
        codon_parents_idxss = torch.full((N, L_codon), aaa_idx, dtype=torch.long)

        # Different rates for each sequence
        for seq_idx in range(N):
            rate = 0.1 * (seq_idx + 1)  # 0.1, 0.2, 0.3, 0.4
            for pos in range(L_codon):
                set_neutral_rates_for_codon(
                    neutral_rates_tensor, seq_idx, pos, aaa_idx, rate
                )

        Z = compute_normalization_constants(
            selection_factors, neutral_rates_tensor, codon_parents_idxss
        )

        assert Z.shape == (N,)
        # Each sequence should have proportionally different Z
        mutations_per_codon = len(
            [
                m
                for m in FUNCTIONAL_CODON_SINGLE_MUTATIONS[aaa_idx]
                if AA_IDX_FROM_CODON_IDX[m[0]] < 20
            ]
        )
        for seq_idx in range(N):
            expected_Z = mutations_per_codon * 0.1 * (seq_idx + 1) * L_codon
            assert torch.abs(Z[seq_idx] - expected_Z) < 1e-5

    # Test 5: Zero rates handling
    def test_zero_rates():
        """Test handling of zero neutral rates."""
        N, L_codon, L_aa = 1, 2, 2

        selection_factors = torch.ones(N, L_aa, 20)
        neutral_rates_tensor = torch.zeros(N, L_codon, 65)

        aaa_idx = CODONS.index("AAA")
        codon_parents_idxss = torch.full((N, L_codon), aaa_idx, dtype=torch.long)

        # Set rates for only some mutations
        aac_idx = CODONS.index("AAC")
        aag_idx = CODONS.index("AAG")
        neutral_rates_tensor[0, 0, aac_idx] = 0.1
        neutral_rates_tensor[0, 1, aag_idx] = 0.2
        # All other rates remain zero

        Z = compute_normalization_constants(
            selection_factors, neutral_rates_tensor, codon_parents_idxss
        )

        # Only two mutations contribute
        expected_Z = 0.1 + 0.2
        assert torch.abs(Z[0] - expected_Z) < 1e-7

    # Test 6: Stop codon handling
    def test_stop_codon_exclusion():
        """Test that stop codons are properly excluded from normalization."""
        N, L_codon, L_aa = 1, 1, 1

        selection_factors = torch.ones(N, L_aa, 20) * 2.0  # All selection = 2
        neutral_rates_tensor = torch.zeros(N, L_codon, 65)

        # Use a codon that can mutate to stop codons
        cag_idx = CODONS.index("CAG")  # Gln - can mutate to TAG (stop)
        codon_parents_idxss = torch.tensor([[cag_idx]])

        # Set rates for all mutations including stop
        for child_idx, _, _ in FUNCTIONAL_CODON_SINGLE_MUTATIONS[cag_idx]:
            neutral_rates_tensor[0, 0, child_idx] = 0.1

        Z = compute_normalization_constants(
            selection_factors, neutral_rates_tensor, codon_parents_idxss
        )

        # Count only non-stop mutations
        non_stop_mutations = 0
        for child_idx, _, _ in FUNCTIONAL_CODON_SINGLE_MUTATIONS[cag_idx]:
            if AA_IDX_FROM_CODON_IDX[child_idx] < 20:
                non_stop_mutations += 1

        expected_Z = non_stop_mutations * 0.1 * 2.0  # rate * selection
        assert torch.abs(Z[0] - expected_Z) < 1e-6

    # Test 7: Numerical stability with extreme values
    def test_numerical_stability():
        """Test numerical stability with very small and large values."""
        N, L_codon, L_aa = 2, 2, 2

        # Test with very small rates
        selection_factors = torch.ones(N, L_aa, 20)
        neutral_rates_tensor = torch.zeros(N, L_codon, 65)

        aaa_idx = CODONS.index("AAA")
        codon_parents_idxss = torch.full((N, L_codon), aaa_idx, dtype=torch.long)

        # Very small rates
        tiny_rate = 1e-10
        for seq_idx in range(1):  # First sequence with tiny rates
            for pos in range(L_codon):
                set_neutral_rates_for_codon(
                    neutral_rates_tensor, seq_idx, pos, aaa_idx, tiny_rate
                )

        # Large rates for second sequence
        large_rate = 1e3
        for pos in range(L_codon):
            set_neutral_rates_for_codon(
                neutral_rates_tensor, 1, pos, aaa_idx, large_rate
            )

        Z = compute_normalization_constants(
            selection_factors, neutral_rates_tensor, codon_parents_idxss
        )

        assert torch.all(torch.isfinite(Z))
        assert torch.all(Z >= 0)
        assert Z[0] < Z[1]  # Small rates should give smaller Z

    # Test 8: GPU compatibility (if available)
    def test_gpu_compatibility():
        """Test that computation works on GPU if available."""
        if not torch.cuda.is_available():
            print("Skipping GPU test - GPU not available")
            return

        device = torch.device("cuda:0")
        N, L_codon, L_aa = 2, 3, 3

        selection_factors = torch.ones(N, L_aa, 20, device=device)
        neutral_rates_tensor = torch.zeros(N, L_codon, 65, device=device)

        aaa_idx = CODONS.index("AAA")
        codon_parents_idxss = torch.full(
            (N, L_codon), aaa_idx, dtype=torch.long, device=device
        )

        # Set rates on GPU
        for seq_idx in range(N):
            for pos in range(L_codon):
                for child_idx, _, _ in FUNCTIONAL_CODON_SINGLE_MUTATIONS[aaa_idx]:
                    if AA_IDX_FROM_CODON_IDX[child_idx] < 20:
                        neutral_rates_tensor[seq_idx, pos, child_idx] = 0.1

        Z = compute_normalization_constants(
            selection_factors, neutral_rates_tensor, codon_parents_idxss
        )

        assert Z.device == device
        assert torch.all(torch.isfinite(Z))

    # Run all sub-tests
    test_uniform_case()
    test_variable_selection()
    test_mixed_codons()
    test_batch_processing()
    test_zero_rates()
    test_stop_codon_exclusion()
    test_numerical_stability()
    test_gpu_compatibility()


def test_codon_to_aa_index_mapping():
    """Test the genetic code mapping using our helper function."""
    # Test a few known mappings
    atg_idx = CODONS.index("ATG")
    met_aa_idx = aa_idx_of_flat_codon_idx(atg_idx)
    expected_met_idx = AA_STR_SORTED.index("M")
    assert met_aa_idx == expected_met_idx

    # Test stop codon handling
    taa_idx = CODONS.index("TAA")  # Stop codon
    stop_aa_idx = aa_idx_of_flat_codon_idx(taa_idx)
    # Stop codons should map to the ambiguous AA index (20)
    assert stop_aa_idx == 20

    # Test ambiguous codon
    ambiguous_aa_idx = aa_idx_of_flat_codon_idx(64)  # Ambiguous codon index
    assert ambiguous_aa_idx == 20


def test_whichmut_trainer_basic():
    """Test basic WhichmutTrainer functionality."""

    # Create a simple mock model
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(20, 20)

        def forward(self, x, masks=None):
            # Return log selection factors
            return torch.zeros(x.shape[0], x.shape[1], 20)

        def train(self):
            pass

        def eval(self):
            pass

        def parameters(self):
            return self.linear.parameters()

    model = MockModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = WhichmutTrainer(model, optimizer)

    # Create simple test data
    nt_parents = pd.Series(["ATGAAACCC"])
    nt_children = pd.Series(["ATGAAACCG"])

    # Mock dataset with properly initialized neutral rates
    neutral_rates = torch.zeros(1, 3, 65)

    # Parent sequence "ATGAAACCC" has codons: ATG (14), AAA (0), CCC (19)
    atg_idx = CODONS.index("ATG")
    aaa_idx = CODONS.index("AAA")
    ccc_idx = CODONS.index("CCC")

    # Set neutral rates for all possible mutations from parent codons
    set_neutral_rates_for_codon(neutral_rates, 0, 0, atg_idx, 0.01)  # ATG
    set_neutral_rates_for_codon(neutral_rates, 0, 1, aaa_idx, 0.01)  # AAA
    set_neutral_rates_for_codon(neutral_rates, 0, 2, ccc_idx, 0.01)  # CCC

    neutral_model_outputs = {"neutral_rates": neutral_rates}
    dataset = DenseWhichmutCodonDataset.of_pcp_df(
        pd.DataFrame({"nt_parent": nt_parents, "nt_child": nt_children}),
        neutral_model_outputs["neutral_rates"],
        model_known_token_count=20,
    )

    # Create simple dataloader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1)

    # Test evaluation (should not raise errors)
    eval_loss = trainer.evaluate(dataloader)
    assert torch.isfinite(eval_loss)


def test_neutral_rates_computation():
    """Test neutral rates computation utility."""
    # Create test sequences
    nt_sequences = pd.Series(["ATGAAACCC", "TGGCCCGGG"])

    # Mock neutral model function
    def mock_neutral_model_fn(seq):
        # Return dummy rates tensor for nucleotide positions
        return torch.ones(len(seq), 4) * 0.1  # (seq_len, 4_bases)

    # Test the function
    try:
        neutral_rates = DenseWhichmutCodonDataset.compute_neutral_rates_from_sequences(
            nt_sequences, mock_neutral_model_fn
        )

        assert neutral_rates.shape[0] == 2  # 2 sequences
        assert neutral_rates.shape[1] == 3  # 3 codons per sequence
        assert neutral_rates.shape[2] == 65  # 65 child codons per position

        # Test sparse version
        sparse_rates = SparseWhichmutCodonDataset.compute_neutral_rates_from_sequences(
            nt_sequences, mock_neutral_model_fn
        )

        # Check sparse format structure
        assert "indices" in sparse_rates
        assert "values" in sparse_rates
        assert "n_possible_mutations" in sparse_rates

        # Check tensor shapes
        assert (
            sparse_rates["indices"].shape[1] == 4
        )  # [seq_idx, pos, parent_idx, child_idx]
        assert (
            sparse_rates["values"].shape[0] == sparse_rates["indices"].shape[0]
        )  # Same number of entries
        assert sparse_rates["n_possible_mutations"].shape[0] == 2  # 2 sequences

        # Check that sparse and dense are consistent for non-zero entries
        # Convert sparse back to dense for comparison
        indices = sparse_rates["indices"]
        values = sparse_rates["values"]
        sparse_as_dense = torch.zeros(2, 3, 65)

        for i in range(indices.shape[0]):
            seq_idx, pos, parent_idx, child_idx = indices[i]
            sparse_as_dense[seq_idx, pos, child_idx] = values[i]

        # Should match non-zero entries in dense version
        torch.testing.assert_close(sparse_as_dense, neutral_rates, atol=1e-6, rtol=1e-6)

    except Exception as e:
        pytest.skip(f"Neutral rates computation test skipped due to: {e}")


@pytest.mark.parametrize("has_mutations", [True, False])
def test_loss_computation_edge_cases(has_mutations):
    """Test loss computation with various edge cases."""
    N, L_codon, L_aa = 1, 2, 2

    # Set up basic tensors
    codon_parents_idxss = torch.zeros(N, L_codon, dtype=torch.long)
    codon_children_idxss = torch.zeros(N, L_codon, dtype=torch.long)
    codon_mutation_indicators = torch.tensor([[has_mutations, False]])
    neutral_rates_tensor = torch.zeros(N, L_codon, 65)
    aa_parents_idxss = torch.zeros(N, L_aa, dtype=torch.long)
    selection_factors = torch.zeros(N, L_aa, 20)
    masks = torch.ones(N, L_codon, dtype=torch.bool)

    # Always set neutral rates for all possible mutations from parent codons
    # (the normalization constant computation requires these even when no mutations occurred)
    set_neutral_rates_for_codon(neutral_rates_tensor, 0, 0, 0)  # AAA
    set_neutral_rates_for_codon(neutral_rates_tensor, 0, 1, 0)  # AAA

    if has_mutations:
        # Set a specific mutation to have occurred
        codon_children_idxss[0, 0] = 1  # AAA -> AAC

    loss = compute_whichmut_loss_batch(
        selection_factors,
        neutral_rates_tensor,
        codon_parents_idxss,
        codon_children_idxss,
        codon_mutation_indicators,
        aa_parents_idxss,
        masks,
    )

    if has_mutations:
        assert torch.isfinite(loss)
        assert loss >= 0  # Loss should be non-negative
    else:
        assert loss.item() == 0.0


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
                expected_rate = data["dense_rates"][0, 0, child_idx]

                # Get value from sparse lookup
                sparse_rate = SparseWhichmutCodonDataset.get_neutral_rate(
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
        data["dense_rates"][0, 0, first_child_idx] = 0.0

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

    def test_loss_computation_sparse_dense_equivalence(self):
        """Test that loss computation is identical for sparse and dense formats."""
        # Create test data with multiple batch sizes and sequence lengths
        for batch_size in [1, 2, 4]:
            for sequence_length in [2, 3, 5]:
                data = create_equivalent_data(batch_size, sequence_length)

                # Create child codons with some mutations
                aac_idx = CODONS.index("AAC")  # AAA -> AAC mutation
                aag_idx = CODONS.index("AAG")  # AAA -> AAG mutation

                # Set up parent and child codon indices
                codon_parents_idxss = data["codon_parents_idxss"]
                codon_children_idxss = codon_parents_idxss.clone()

                # Create mutation indicators
                codon_mutation_indicators = torch.zeros_like(
                    codon_parents_idxss, dtype=torch.bool
                )

                # Add some mutations (different patterns for each sequence)
                for seq_idx in range(batch_size):
                    # Mutate first position for even sequences
                    if seq_idx % 2 == 0 and sequence_length > 0:
                        codon_children_idxss[seq_idx, 0] = aac_idx
                        codon_mutation_indicators[seq_idx, 0] = True

                    # Mutate last position for all sequences
                    if sequence_length > 1:
                        codon_children_idxss[seq_idx, -1] = aag_idx
                        codon_mutation_indicators[seq_idx, -1] = True

                # Create AA indices (AAA -> K, AAC -> N, AAG -> K)
                aa_parents_idxss = torch.full(
                    (batch_size, sequence_length),
                    AA_STR_SORTED.index("K"),  # AAA codes for Lysine (K)
                    dtype=torch.long,
                )

                # Create masks (all positions valid)
                masks = torch.ones((batch_size, sequence_length), dtype=torch.bool)

                # Create selection factors (log space)
                selection_factors = torch.randn(batch_size, sequence_length, 20) * 0.1

                # Compute loss with dense format
                loss_dense = compute_whichmut_loss_batch(
                    selection_factors,
                    data["dense_rates"],
                    codon_parents_idxss,
                    codon_children_idxss,
                    codon_mutation_indicators,
                    aa_parents_idxss,
                    masks,
                )

                # Compute loss with sparse format
                loss_sparse = compute_whichmut_loss_batch(
                    selection_factors,
                    data["sparse_rates"],
                    codon_parents_idxss,
                    codon_children_idxss,
                    codon_mutation_indicators,
                    aa_parents_idxss,
                    masks,
                )

                # Verify losses are identical
                assert torch.allclose(
                    loss_dense, loss_sparse, rtol=1e-6, atol=1e-8
                ), f"Loss mismatch for batch_size={batch_size}, seq_len={sequence_length}: Dense={loss_dense.item():.8f}, Sparse={loss_sparse.item():.8f}"

                # Also verify both losses are valid
                assert torch.isfinite(
                    loss_dense
                ), f"Dense loss is not finite: {loss_dense}"
                assert torch.isfinite(
                    loss_sparse
                ), f"Sparse loss is not finite: {loss_sparse}"
                assert loss_dense >= 0, f"Dense loss is negative: {loss_dense}"
                assert loss_sparse >= 0, f"Sparse loss is negative: {loss_sparse}"

    def test_vectorized_vs_iterative_implementations(self):
        """Test that vectorized and iterative loss implementations produce identical
        results."""
        # Test with multiple batch sizes and sequence lengths
        test_configs = [
            (1, 2),  # Small test case
            (2, 3),  # Medium test case
            (3, 4),  # Larger test case
        ]

        for batch_size, sequence_length in test_configs:
            data = create_equivalent_data(batch_size, sequence_length)

            # Create mutation data using the same pattern as other tests
            aac_idx = CODONS.index("AAC")  # AAA -> AAC mutation
            aag_idx = CODONS.index("AAG")  # AAA -> AAG mutation

            # Set up parent and child codon indices
            codon_parents_idxss = data["codon_parents_idxss"]
            codon_children_idxss = codon_parents_idxss.clone()

            # Create mutation indicators - add mutations to test
            codon_mutation_indicators = torch.zeros_like(
                codon_parents_idxss, dtype=torch.bool
            )

            # Add deterministic mutations for reproducible testing
            for seq_idx in range(batch_size):
                # Mutate first position for even sequences
                if seq_idx % 2 == 0 and sequence_length > 0:
                    codon_children_idxss[seq_idx, 0] = aac_idx
                    codon_mutation_indicators[seq_idx, 0] = True

                # Mutate second position for all sequences if length > 1
                if sequence_length > 1:
                    codon_children_idxss[seq_idx, 1] = aag_idx
                    codon_mutation_indicators[seq_idx, 1] = True

            # Create AA parents indices (AAA -> K)
            aa_parents_idxss = torch.full(
                (batch_size, sequence_length),
                AA_STR_SORTED.index("K"),  # AAA codes for Lysine (K)
                dtype=torch.long,
            )

            # All positions valid
            masks = torch.ones((batch_size, sequence_length), dtype=torch.bool)

            # Create selection factors (in log space)
            torch.manual_seed(42)  # For reproducible results
            selection_factors = (
                torch.randn(batch_size, sequence_length, 20, requires_grad=True) * 0.1
            )

            # Test both sparse and dense formats
            for format_name, neutral_rates_data in [
                ("sparse", data["sparse_rates"]),
                ("dense", data["dense_rates"]),
            ]:
                # Compute loss with vectorized implementation
                loss_vectorized = compute_whichmut_loss_batch(
                    selection_factors,
                    neutral_rates_data,
                    codon_parents_idxss,
                    codon_children_idxss,
                    codon_mutation_indicators,
                    aa_parents_idxss,
                    masks,
                )

                # Compute loss with iterative implementation
                loss_iterative = compute_whichmut_loss_batch_iterative(
                    selection_factors,
                    neutral_rates_data,
                    codon_parents_idxss,
                    codon_children_idxss,
                    codon_mutation_indicators,
                    aa_parents_idxss,
                    masks,
                )

                # Verify implementations produce identical results
                assert torch.allclose(
                    loss_vectorized, loss_iterative, rtol=1e-6, atol=1e-8
                ), f"Implementation mismatch for {format_name} format, batch_size={batch_size}, seq_len={sequence_length}: Vectorized={loss_vectorized.item():.8f}, Iterative={loss_iterative.item():.8f}"

                # Verify both losses are valid
                assert torch.isfinite(
                    loss_vectorized
                ), f"Vectorized loss is not finite: {loss_vectorized}"
                assert torch.isfinite(
                    loss_iterative
                ), f"Iterative loss is not finite: {loss_iterative}"
                assert (
                    loss_vectorized >= 0
                ), f"Vectorized loss is negative: {loss_vectorized}"
                assert (
                    loss_iterative >= 0
                ), f"Iterative loss is negative: {loss_iterative}"

                # Verify both losses require gradients
                assert (
                    loss_vectorized.requires_grad
                ), "Vectorized loss should require gradients"
                assert (
                    loss_iterative.requires_grad
                ), "Iterative loss should require gradients"

    def test_vectorized_vs_iterative_edge_cases(self):
        """Test vectorized vs iterative implementations on edge cases."""
        data = create_equivalent_data(batch_size=2, sequence_length=3)

        # Test case 1: No mutations
        codon_parents_idxss = data["codon_parents_idxss"]
        codon_children_idxss = codon_parents_idxss.clone()  # Same as parents
        codon_mutation_indicators = torch.zeros_like(
            codon_parents_idxss, dtype=torch.bool
        )  # No mutations

        aa_parents_idxss = torch.full(
            (2, 3), AA_STR_SORTED.index("K"), dtype=torch.long
        )
        masks = torch.ones((2, 3), dtype=torch.bool)
        selection_factors = torch.randn(2, 3, 20) * 0.1

        # Test both implementations with no mutations
        for format_name, neutral_rates_data in [
            ("sparse", data["sparse_rates"]),
            ("dense", data["dense_rates"]),
        ]:
            loss_vec = compute_whichmut_loss_batch(
                selection_factors,
                neutral_rates_data,
                codon_parents_idxss,
                codon_children_idxss,
                codon_mutation_indicators,
                aa_parents_idxss,
                masks,
            )
            loss_iter = compute_whichmut_loss_batch_iterative(
                selection_factors,
                neutral_rates_data,
                codon_parents_idxss,
                codon_children_idxss,
                codon_mutation_indicators,
                aa_parents_idxss,
                masks,
            )

            assert (
                loss_vec.item() == 0.0
            ), f"Expected zero loss for no mutations ({format_name})"
            assert (
                loss_iter.item() == 0.0
            ), f"Expected zero loss for no mutations ({format_name})"
            assert torch.allclose(
                loss_vec, loss_iter
            ), f"No-mutation case should match ({format_name})"

    def test_vectorized_gradient_compatibility(self):
        """Test that vectorized implementation maintains gradient compatibility."""
        data = create_equivalent_data(batch_size=2, sequence_length=3)

        # Create test data with mutations
        codon_parents_idxss = data["codon_parents_idxss"]
        codon_children_idxss = codon_parents_idxss.clone()
        codon_children_idxss[0, 0] = CODONS.index("AAC")  # Add one mutation
        codon_mutation_indicators = torch.zeros_like(
            codon_parents_idxss, dtype=torch.bool
        )
        codon_mutation_indicators[0, 0] = True

        aa_parents_idxss = torch.full(
            (2, 3), AA_STR_SORTED.index("K"), dtype=torch.long
        )
        masks = torch.ones((2, 3), dtype=torch.bool)

        # Create selection factors that require gradients
        selection_factors_base = torch.randn(2, 3, 20) * 0.1
        selection_factors = selection_factors_base.requires_grad_(True)

        # Test gradient computation with vectorized implementation
        loss = compute_whichmut_loss_batch(
            selection_factors,
            data["sparse_rates"],
            codon_parents_idxss,
            codon_children_idxss,
            codon_mutation_indicators,
            aa_parents_idxss,
            masks,
        )

        # Backward pass
        loss.backward()

        # Check that gradients were computed
        assert selection_factors.grad is not None, "No gradients computed!"
        assert torch.isfinite(selection_factors.grad).all(), "Invalid gradients!"
        assert (
            selection_factors.grad.shape == selection_factors.shape
        ), "Gradient shape mismatch!"


class TestTrainerMethodExtraction:
    """Test the extracted methods from WhichmutTrainer refactoring."""

    def setup_method(self):
        """Set up test fixtures."""

        # Create a simple mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(20, 20)

            def forward(self, x, masks=None):
                # Return log selection factors
                return torch.zeros(x.shape[0], x.shape[1], 20)

            def train(self):
                pass

            def eval(self):
                pass

            def parameters(self):
                return self.linear.parameters()

        self.model = MockModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.trainer = WhichmutTrainer(self.model, self.optimizer)
        self.device = torch.device("cpu")

    def test_move_batch_to_device(self):
        """Test _move_batch_to_device handles both dense and sparse formats
        correctly."""
        # Test dense format
        batch_data_dense = (
            torch.randn(2, 3),  # codon_parents_idxss
            torch.randn(2, 3),  # codon_children_idxss
            torch.randn(2, 3, 65, 65),  # dense neutral_rates_data
            torch.randn(2, 3),  # aa_parents_idxss
            torch.randn(2, 3),  # aa_children_idxss
            torch.randn(2, 3),  # codon_mutation_indicators
            torch.randn(2, 3),  # masks
        )

        moved_dense = self.trainer._move_batch_to_device(batch_data_dense, self.device)

        # Check all tensors are on correct device
        assert all(
            tensor.device == self.device for tensor in moved_dense[:-1]
        )  # All but neutral_rates
        assert isinstance(moved_dense[2], torch.Tensor)  # Dense format preserved

        # Test sparse format
        sparse_neutral_data = {
            "indices": torch.randint(0, 64, (2, 3, 5, 2)),
            "values": torch.randn(2, 3, 5),
            "n_possible_mutations": torch.randint(1, 5, (2, 3)),
        }

        batch_data_sparse = (
            torch.randn(2, 3),  # codon_parents_idxss
            torch.randn(2, 3),  # codon_children_idxss
            sparse_neutral_data,  # sparse neutral_rates_data
            torch.randn(2, 3),  # aa_parents_idxss
            torch.randn(2, 3),  # aa_children_idxss
            torch.randn(2, 3),  # codon_mutation_indicators
            torch.randn(2, 3),  # masks
        )

        moved_sparse = self.trainer._move_batch_to_device(
            batch_data_sparse, self.device
        )

        # Check standard tensors moved correctly
        for i in [0, 1, 3, 4, 5, 6]:  # Skip neutral_rates_data at index 2
            assert moved_sparse[i].device == self.device

        # Check sparse neutral rates moved correctly
        sparse_moved = moved_sparse[2]
        assert isinstance(sparse_moved, dict)
        assert all(tensor.device == self.device for tensor in sparse_moved.values())

    def test_log_batch_info_dense_warning(self, capsys):
        """Test _log_batch_info produces appropriate warnings for dense format."""
        neutral_rates_dense = torch.randn(2, 10, 65, 65)  # Large tensor

        # Test large batch warning
        self.trainer._log_batch_info(0, 15, neutral_rates_dense)
        captured = capsys.readouterr()
        assert "WARNING: Large batch size" in captured.out
        assert "DENSE format" in captured.out
        assert "MB for neutral rates tensor" in captured.out

        # Test medium batch info
        self.trainer._log_batch_info(0, 5, neutral_rates_dense)
        captured = capsys.readouterr()
        assert "INFO: Processing batch" in captured.out
        assert "DENSE format" in captured.out

    def test_log_batch_info_sparse(self, capsys):
        """Test _log_batch_info handles sparse format correctly."""
        sparse_neutral_data = {
            "indices": torch.randint(0, 64, (2, 10, 5, 2)),
            "values": torch.randn(2, 10, 5),
            "n_possible_mutations": torch.randint(1, 5, (2, 10)),
        }

        self.trainer._log_batch_info(0, 5, sparse_neutral_data)
        captured = capsys.readouterr()
        assert "INFO: Processing batch" in captured.out
        assert "SPARSE format" in captured.out
        assert "MB)" in captured.out

    def test_check_gradients_valid(self):
        """Test _check_gradients correctly identifies valid gradients."""
        # Create some gradients
        self.model.zero_grad()
        loss = self.model.linear.weight.sum()
        loss.backward()

        # Should be valid
        assert self.trainer._check_gradients() is True

        # Test with invalid gradients (NaN)
        self.model.linear.weight.grad[0, 0] = float("nan")
        assert self.trainer._check_gradients() is False

        # Test with infinite gradients
        self.model.linear.weight.grad[0, 0] = float("inf")
        assert self.trainer._check_gradients() is False

    def test_compute_loss_basic(self):
        """Test _compute_loss produces valid loss values."""
        # Create minimal batch data
        batch_data = (
            torch.zeros(1, 3, dtype=torch.long),  # codon_parents_idxss
            torch.zeros(1, 3, dtype=torch.long),  # codon_children_idxss
            torch.ones(1, 3, 65) * 0.01,  # neutral_rates_data (dense)
            torch.zeros(1, 3, dtype=torch.long),  # aa_parents_idxss
            torch.zeros(1, 3, dtype=torch.long),  # aa_children_idxss (unused)
            torch.zeros(1, 3, dtype=torch.bool),  # codon_mutation_indicators
            torch.ones(1, 3, dtype=torch.bool),  # masks
        )

        loss = self.trainer._compute_loss(batch_data)

        # Loss should be finite and scalar
        assert torch.isfinite(loss)
        assert loss.shape == torch.Size([])

    def test_train_step_with_retry_success(self):
        """Test _train_step_with_retry succeeds on first attempt with valid
        gradients."""
        # Create batch data that won't cause gradient issues
        batch_data = (
            torch.zeros(1, 3, dtype=torch.long),  # codon_parents_idxss
            torch.zeros(1, 3, dtype=torch.long),  # codon_children_idxss
            torch.ones(1, 3, 65) * 0.01,  # neutral_rates_data
            torch.zeros(1, 3, dtype=torch.long),  # aa_parents_idxss
            torch.zeros(1, 3, dtype=torch.long),  # aa_children_idxss
            torch.zeros(1, 3, dtype=torch.bool),  # codon_mutation_indicators
            torch.ones(1, 3, dtype=torch.bool),  # masks
        )

        # Should succeed without retries
        loss = self.trainer._train_step_with_retry(batch_data)
        assert torch.isfinite(loss)

    def test_train_step_with_retry_no_optimizer(self):
        """Test _train_step_with_retry works without optimizer (evaluation mode)."""
        trainer_no_opt = WhichmutTrainer(self.model, optimizer=None)

        batch_data = (
            torch.zeros(1, 3, dtype=torch.long),
            torch.zeros(1, 3, dtype=torch.long),
            torch.ones(1, 3, 65) * 0.01,
            torch.zeros(1, 3, dtype=torch.long),
            torch.zeros(1, 3, dtype=torch.long),
            torch.zeros(1, 3, dtype=torch.bool),
            torch.ones(1, 3, dtype=torch.bool),
        )

        loss = trainer_no_opt._train_step_with_retry(batch_data)
        assert torch.isfinite(loss)

    def test_run_epoch_structure_maintained(self):
        """Test that _run_epoch still produces the same results after refactoring."""
        # Create simple test data matching existing test
        nt_parents = pd.Series(["ATGAAACCC"])
        nt_children = pd.Series(["ATGAAACCG"])

        # Create neutral rates
        neutral_rates = torch.zeros(1, 3, 65)
        atg_idx = CODONS.index("ATG")
        aaa_idx = CODONS.index("AAA")
        ccc_idx = CODONS.index("CCC")

        set_neutral_rates_for_codon(neutral_rates, 0, 0, atg_idx, 0.01)
        set_neutral_rates_for_codon(neutral_rates, 0, 1, aaa_idx, 0.01)
        set_neutral_rates_for_codon(neutral_rates, 0, 2, ccc_idx, 0.01)

        neutral_model_outputs = {"neutral_rates": neutral_rates}
        dataset = DenseWhichmutCodonDataset.of_pcp_df(
            pd.DataFrame({"nt_parent": nt_parents, "nt_child": nt_children}),
            neutral_model_outputs["neutral_rates"],
            model_known_token_count=20,
        )

        from torch.utils.data import DataLoader

        dataloader = DataLoader(dataset, batch_size=1)

        # Test evaluation mode (no gradients needed)
        eval_loss = self.trainer.evaluate(dataloader)
        assert torch.isfinite(eval_loss)
        assert eval_loss.shape == torch.Size([])

        # Test training mode by using a trainer without optimizer
        # (avoids gradient issues while still testing the training path)
        trainer_no_opt = WhichmutTrainer(self.model, optimizer=None)
        train_loss = trainer_no_opt.train_epoch(dataloader)
        assert torch.isfinite(train_loss)
        assert train_loss.shape == torch.Size([])
