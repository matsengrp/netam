import numpy as np
import pandas as pd
import torch
from typing import Tuple  # , Dict, List

from Bio.Data import CodonTable
from netam.sequences import (
    AA_STR_SORTED,
    AMBIGUOUS_CODON_IDX,
    CODONS,
    STOP_CODONS,
    # aa_index_of_codon,
    translate_sequences,
)
from netam.common import BIG


def single_mutant_aa_indices(codon):
    """Given a codon, return the amino acid indices for all single-mutant neighbors.

    Args:
        codon (str): A three-letter codon (e.g., "ATG").
        AA_STR_SORTED (str): A string of amino acids in a sorted order.

    Returns:
        list of int: Indices of the resulting amino acids for single mutants.
    """
    standard_table = CodonTable.unambiguous_dna_by_id[1]  # Standard codon table
    bases = ["A", "C", "G", "T"]

    mutant_aa_indices = set()  # Use a set to avoid duplicates

    # Generate all single-mutant neighbors
    for pos in range(3):  # Codons have 3 positions
        for base in bases:
            if base != codon[pos]:  # Mutate only if it's a different base
                mutant_codon = codon[:pos] + base + codon[pos + 1 :]

                # Check if the mutant codon translates to a valid amino acid
                if mutant_codon in standard_table.forward_table:
                    mutant_aa = standard_table.forward_table[mutant_codon]
                    mutant_aa_indices.add(AA_STR_SORTED.index(mutant_aa))

    return sorted(mutant_aa_indices)


def make_codon_neighbor_indicator(nt_seq):
    """Create a binary array indicating the single-mutant amino acid neighbors of each
    codon in a given DNA sequence."""
    neighbor = np.zeros((len(AA_STR_SORTED), len(nt_seq) // 3), dtype=bool)
    for i in range(0, len(nt_seq), 3):
        codon = nt_seq[i : i + 3]
        neighbor[single_mutant_aa_indices(codon), i // 3] = True
    return neighbor


def generate_codon_aa_indicator_matrix():
    """Generate a matrix that maps codons (rows) to amino acids (columns)."""

    matrix = np.zeros((len(CODONS), len(AA_STR_SORTED)))

    for i, codon in enumerate(CODONS):
        try:
            aa = translate_sequences([codon])[0]
        except ValueError:  # Handle STOP codon
            pass
        else:
            aa_idx = AA_STR_SORTED.index(aa)
            matrix[i, aa_idx] = 1

    return matrix


CODON_AA_INDICATOR_MATRIX = torch.tensor(
    generate_codon_aa_indicator_matrix(), dtype=torch.float32
)


def build_stop_codon_indicator_tensor():
    """Return a tensor indicating the stop codons."""
    stop_codon_indicator = torch.zeros(len(CODONS))
    for stop_codon in STOP_CODONS:
        stop_codon_indicator[CODONS.index(stop_codon)] = 1.0
    return stop_codon_indicator


STOP_CODON_INDICATOR = build_stop_codon_indicator_tensor()

STOP_CODON_ZAPPER = STOP_CODON_INDICATOR * -BIG

# We build a table that will allow us to look up the amino acid index
# from the codon indices. Argmax gets the aa index.
AA_IDX_FROM_CODON = CODON_AA_INDICATOR_MATRIX.argmax(dim=1).view(4, 4, 4)


def aa_idxs_of_codon_idxs(codon_idx_tensor):
    """Translate an unflattened codon index tensor of shape (L, 3) to a tensor of amino
    acid indices."""
    # Get the amino acid index for each parent codon.
    return AA_IDX_FROM_CODON[
        (
            codon_idx_tensor[:, 0],
            codon_idx_tensor[:, 1],
            codon_idx_tensor[:, 2],
        )
    ]


def generate_codon_neighbor_matrix():
    """Generate codon neighbor matrix for efficient single-mutation lookups.

    Returns:
        torch.Tensor: A (65, 20) boolean matrix where entry (i, j) is True if
                     codon i can mutate to amino acid j via single nucleotide substitution.
                     Row 64 (AMBIGUOUS_CODON_IDX) will be all False.
    """
    # Include space for ambiguous codon at index 64
    matrix = np.zeros((AMBIGUOUS_CODON_IDX + 1, len(AA_STR_SORTED)), dtype=bool)

    # Only process the 64 standard codons, not the ambiguous codon
    for i, codon in enumerate(CODONS):
        mutant_aa_indices = single_mutant_aa_indices(codon)
        matrix[i, mutant_aa_indices] = True

    # Row 64 (AMBIGUOUS_CODON_IDX) remains all False

    return torch.tensor(matrix, dtype=torch.bool)


def generate_codon_single_mutation_map():
    """Generate mapping of codon-to-codon single mutations.

    Returns:
        Dict[int, List[Tuple[int, int, str]]]: Maps parent codon index to list of
        (child_codon_idx, nt_position, new_base) for all single mutations.
        Only includes valid codons (0-63), not AMBIGUOUS_CODON_IDX (64).
    """
    mutation_map = {}

    # Only process the 64 valid codons, not the ambiguous codon at index 64
    for parent_idx, parent_codon in enumerate(CODONS):
        mutations = []
        for nt_pos in range(3):
            for new_base in ["A", "C", "G", "T"]:
                if new_base != parent_codon[nt_pos]:
                    child_codon = (
                        parent_codon[:nt_pos] + new_base + parent_codon[nt_pos + 1 :]
                    )
                    child_idx = CODONS.index(child_codon)
                    mutations.append((child_idx, nt_pos, new_base))
        mutation_map[parent_idx] = mutations

    return mutation_map


# Global tensors/mappings for efficient lookups
CODON_NEIGHBOR_MATRIX = generate_codon_neighbor_matrix()  # (65, 20)
CODON_SINGLE_MUTATIONS = generate_codon_single_mutation_map()


def encode_codon_mutations(
    nt_parents: pd.Series, nt_children: pd.Series
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert parent/child nucleotide sequences to codon indices and mutation indicators.

    Args:
        nt_parents: Parent nucleotide sequences
        nt_children: Child nucleotide sequences

    Returns:
        Tuple of:
        - codon_parents_idxss: (N, L_codon) tensor of parent codon indices
        - codon_children_idxss: (N, L_codon) tensor of child codon indices
        - codon_mutation_indicators: (N, L_codon) boolean tensor indicating mutation positions
    """
    # Implementation will use existing netam functions:
    # - encode_sequences() for converting to indices
    # - Compare parent vs child codon indices to identify mutations
    pass


def create_codon_masks(nt_parents: pd.Series, nt_children: pd.Series) -> torch.Tensor:
    """Create masks for valid codon positions, masking ambiguous codons (containing Ns).

    Args:
        nt_parents: Parent nucleotide sequences
        nt_children: Child nucleotide sequences

    Returns:
        masks: (N, L_codon) boolean tensor indicating valid codon positions

    Raises:
        ValueError: If any sequences contain stop codons
    """
    # Implementation will:
    # - Assert no stop codons in any sequences (halt with clear error)
    # - Check sequence lengths are multiples of 3
    # - Mask positions with ambiguous codons (containing N) using AMBIGUOUS_CODON_IDX
    # - Use existing netam masking patterns
    pass
