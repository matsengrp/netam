import torch
from netam.codon_table import (
    generate_codon_neighbor_matrix,
    generate_codon_single_mutation_map,
    CODON_NEIGHBOR_MATRIX,
    CODON_SINGLE_MUTATIONS,
)
from netam.sequences import (
    CODONS,
    AA_STR_SORTED,
    AMBIGUOUS_CODON_IDX,
    aa_index_of_codon,
)


def test_generate_codon_neighbor_matrix():
    """Test codon neighbor matrix generation."""
    matrix = generate_codon_neighbor_matrix()

    # Check dimensions
    assert matrix.shape == (65, 20)

    # Check that ambiguous codon row (64) is all False
    assert not matrix[AMBIGUOUS_CODON_IDX].any()

    # Test specific known cases
    # ATG (Met) -> can mutate to Ile, Thr, Arg, Ser, etc.
    atg_idx = CODONS.index("ATG")
    met_idx = AA_STR_SORTED.index("M")
    ile_idx = AA_STR_SORTED.index("I")  # ATG->ATT

    # ATG itself codes for Met, should not be in neighbor matrix
    assert not matrix[atg_idx, met_idx]
    # ATG can mutate to Ile via ATG->ATT
    assert matrix[atg_idx, ile_idx]

    # All valid codons should have at least one possible mutation target
    for i in range(64):  # Only check valid codons
        assert matrix[i].any(), f"Codon {CODONS[i]} has no mutation targets"


def test_generate_codon_single_mutation_map():
    """Test codon single mutation mapping."""
    mutation_map = generate_codon_single_mutation_map()

    # Check all 64 valid codons are present
    assert len(mutation_map) == 64
    assert all(i in mutation_map for i in range(64))

    # Test specific case: ATG
    atg_idx = CODONS.index("ATG")
    atg_mutations = mutation_map[atg_idx]

    # ATG has 9 possible single mutations (3 positions × 3 bases each)
    assert len(atg_mutations) == 9

    # Check format: (child_codon_idx, nt_position, new_base)
    for child_idx, nt_pos, new_base in atg_mutations:
        assert isinstance(child_idx, int)
        assert 0 <= child_idx < 64
        assert nt_pos in [0, 1, 2]
        assert new_base in ["A", "C", "G", "T"]
        assert new_base != "ATG"[nt_pos]  # Should be different from original

        # Verify the mutation is correct
        original_codon = "ATG"
        expected_codon = (
            original_codon[:nt_pos] + new_base + original_codon[nt_pos + 1 :]
        )
        actual_codon = CODONS[child_idx]
        assert actual_codon == expected_codon

    # Test mutation targets are unique
    child_indices = [child_idx for child_idx, _, _ in atg_mutations]
    assert len(child_indices) == len(set(child_indices))


def test_codon_matrices_consistency():
    """Test that neighbor matrix and mutation map are consistent."""
    from netam.sequences import STOP_CODONS

    # For each codon, check that mutation map matches neighbor matrix
    for parent_idx in range(64):
        mutations = CODON_SINGLE_MUTATIONS[parent_idx]

        # Get all amino acids reachable via single mutations
        reachable_aas = set()
        for child_idx, _, _ in mutations:
            child_codon = CODONS[child_idx]
            # Skip stop codons
            if child_codon in STOP_CODONS:
                continue
            child_aa_idx = aa_index_of_codon(child_codon)
            reachable_aas.add(child_aa_idx)

        # Check against neighbor matrix
        neighbor_aas = set(torch.where(CODON_NEIGHBOR_MATRIX[parent_idx])[0].tolist())

        assert (
            reachable_aas == neighbor_aas
        ), f"Mismatch for codon {CODONS[parent_idx]}: {reachable_aas} vs {neighbor_aas}"


def test_codon_utilities_imported():
    """Test that utilities are properly imported."""
    # Check types and basic properties
    assert isinstance(CODON_NEIGHBOR_MATRIX, torch.Tensor)
    assert isinstance(CODON_SINGLE_MUTATIONS, dict)
    assert CODON_NEIGHBOR_MATRIX.shape == (65, 20)
    assert len(CODON_SINGLE_MUTATIONS) == 64


def test_encode_codon_mutations():
    """Test codon mutation encoding."""
    # TODO: Implement test for encode_codon_mutations function
    pass


def test_create_codon_masks():
    """Test codon mask creation."""
    # TODO: Implement test for create_codon_masks function
    pass
