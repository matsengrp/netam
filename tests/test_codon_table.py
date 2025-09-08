import torch
from netam.codon_table import (
    generate_codon_neighbor_matrix,
    generate_codon_single_mutation_map,
    CODON_NEIGHBOR_MATRIX,
    CODON_SINGLE_MUTATIONS,
    AA_IDX_FROM_CODON_IDX,
    encode_codon_mutations,
    create_codon_masks,
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


def test_aa_idx_from_codon_idx_mapping():
    """Test AA_IDX_FROM_CODON_IDX mapping correctness."""
    from netam.sequences import translate_sequences, STOP_CODONS

    # Test that the mapping contains entries for non-stop codons
    assert len(AA_IDX_FROM_CODON_IDX) > 0

    # Test specific known mappings
    atg_idx = CODONS.index("ATG")  # Methionine
    assert AA_IDX_FROM_CODON_IDX[atg_idx] == AA_STR_SORTED.index("M")

    tgg_idx = CODONS.index("TGG")  # Tryptophan
    assert AA_IDX_FROM_CODON_IDX[tgg_idx] == AA_STR_SORTED.index("W")

    # Test that stop codons are not in the mapping
    for stop_codon in STOP_CODONS:
        stop_codon_idx = CODONS.index(stop_codon)
        assert stop_codon_idx not in AA_IDX_FROM_CODON_IDX

    # Test that all non-stop codons are correctly mapped
    for codon_idx, codon in enumerate(CODONS):
        if codon not in STOP_CODONS:
            # Should be in the mapping
            assert codon_idx in AA_IDX_FROM_CODON_IDX
            # Should map to correct AA index
            expected_aa = translate_sequences([codon])[0]
            expected_aa_idx = AA_STR_SORTED.index(expected_aa)
            assert AA_IDX_FROM_CODON_IDX[codon_idx] == expected_aa_idx

    # Test that all AA indices in the mapping are valid
    for codon_idx, aa_idx in AA_IDX_FROM_CODON_IDX.items():
        assert 0 <= aa_idx < len(AA_STR_SORTED)
        assert 0 <= codon_idx < len(CODONS)


def test_codon_utilities_imported():
    """Test that utilities are properly imported."""
    # Check types and basic properties
    assert isinstance(CODON_NEIGHBOR_MATRIX, torch.Tensor)
    assert isinstance(CODON_SINGLE_MUTATIONS, dict)
    assert CODON_NEIGHBOR_MATRIX.shape == (65, 20)
    assert len(CODON_SINGLE_MUTATIONS) == 64


def test_encode_codon_mutations():
    """Test codon mutation encoding."""
    import pandas as pd

    # Test data: simple sequences
    parent_seqs = pd.Series(["ATGAAACCC", "CCCGGGTTT"])
    child_seqs = pd.Series(["ATGAAACCG", "CCCGGGTTT"])  # One mutation in first seq

    parent_indices, child_indices, mutation_indicators = encode_codon_mutations(
        parent_seqs, child_seqs
    )

    # Check shapes
    assert parent_indices.shape == (2, 3)  # 2 sequences, 3 codons each
    assert child_indices.shape == (2, 3)
    assert mutation_indicators.shape == (2, 3)

    # Check first sequence: ATG AAA CCC -> ATG AAA CCG (mutation in 3rd codon)
    assert parent_indices[0, 0] == CODONS.index("ATG")
    assert parent_indices[0, 1] == CODONS.index("AAA")
    assert parent_indices[0, 2] == CODONS.index("CCC")

    assert child_indices[0, 0] == CODONS.index("ATG")
    assert child_indices[0, 1] == CODONS.index("AAA")
    assert child_indices[0, 2] == CODONS.index("CCG")

    # Check mutation indicators
    assert not mutation_indicators[0, 0]  # No mutation
    assert not mutation_indicators[0, 1]  # No mutation
    assert mutation_indicators[0, 2]  # Mutation here

    # Check second sequence: no mutations
    assert not mutation_indicators[1, :].any()


def test_encode_codon_mutations_with_ambiguous():
    """Test codon mutation encoding with ambiguous codons."""
    import pandas as pd

    # Test data with N's
    parent_seqs = pd.Series(["ATGNNNCCG"])
    child_seqs = pd.Series(["ATGNNNCCG"])

    parent_indices, child_indices, mutation_indicators = encode_codon_mutations(
        parent_seqs, child_seqs
    )

    # Check that middle codon gets AMBIGUOUS_CODON_IDX
    assert parent_indices[0, 0] == CODONS.index("ATG")
    assert parent_indices[0, 1] == AMBIGUOUS_CODON_IDX  # NNN -> ambiguous
    assert parent_indices[0, 2] == CODONS.index("CCG")

    assert child_indices[0, 1] == AMBIGUOUS_CODON_IDX


def test_encode_codon_mutations_errors():
    """Test error conditions for encode_codon_mutations."""
    import pandas as pd

    # Different length sequences
    parent_seqs = pd.Series(["ATGAAA"])
    child_seqs = pd.Series(["ATGAAACCC"])

    try:
        encode_codon_mutations(parent_seqs, child_seqs)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "same length" in str(e)

    # Non-multiple of 3
    parent_seqs = pd.Series(["ATGAA"])
    child_seqs = pd.Series(["ATGAA"])

    try:
        encode_codon_mutations(parent_seqs, child_seqs)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "multiple of 3" in str(e)


def test_create_codon_masks():
    """Test codon mask creation."""
    import pandas as pd

    # Test data: normal sequences
    parent_seqs = pd.Series(["ATGAAACCC", "CCCGGGTTT"])
    child_seqs = pd.Series(["ATGAAACCG", "CCCGGGTTT"])

    masks = create_codon_masks(parent_seqs, child_seqs)

    # Check shape
    assert masks.shape == (2, 3)  # 2 sequences, 3 codons each

    # All positions should be valid (True)
    assert masks.all()


def test_create_codon_masks_with_ambiguous():
    """Test codon mask creation with ambiguous codons."""
    import pandas as pd

    # Test data with N's
    parent_seqs = pd.Series(["ATGNNNCCG", "CCCGGGTTT"])
    child_seqs = pd.Series(["ATGNNNCCG", "CCCGGGTNC"])  # N in child too

    masks = create_codon_masks(parent_seqs, child_seqs)

    # First sequence: middle codon should be masked
    assert masks[0, 0]  # ATG - valid
    assert not masks[0, 1]  # NNN - masked
    assert masks[0, 2]  # CCG - valid

    # Second sequence: last codon should be masked
    assert masks[1, 0]  # CCC - valid
    assert masks[1, 1]  # GGG - valid
    assert not masks[1, 2]  # TNC - masked due to N


def test_create_codon_masks_stop_codon_error():
    """Test that stop codons raise an error."""
    import pandas as pd

    # Test with stop codon
    parent_seqs = pd.Series(["ATGTAA"])  # TAA is stop codon
    child_seqs = pd.Series(["ATGTAA"])

    try:
        create_codon_masks(parent_seqs, child_seqs)
        assert False, "Should have raised ValueError for stop codon"
    except ValueError as e:
        assert "stop codon" in str(e)
