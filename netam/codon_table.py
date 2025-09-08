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
    contains_stop_codon,
    idx_of_codon_allowing_ambiguous,
    iter_codons,
    translate_sequences,
    CODON_TO_INDEX,
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


# iterate through codons,
# tranlate to amino acids, and build a mapping
# from codon index to amino acid index.
# This is used to convert codon indices to amino acid indices.
def iter_codon_aa_indices():
    """Yield tuples of (codon_index, amino_acid_index) for all codons."""

    for codon, codon_idx in CODON_TO_INDEX.items():
        try:
            aa = translate_sequences([codon])[0]
            aa_idx = AA_STR_SORTED.index(aa)
            yield codon_idx, aa_idx
        except ValueError:  # Handle STOP codon
            continue


AA_IDX_FROM_CODON_IDX = {
    codon_idx: aa_idx for codon_idx, aa_idx in iter_codon_aa_indices()
}


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


STOP_CODON_IDXS = [CODON_TO_INDEX[codon] for codon in STOP_CODONS]
# is a mapping from parent codon index to a list of tuples
# (child_codon_idx, nt_position, new_base)
# for all single mutations except those which result in stop codons.
FUNCTIONAL_CODON_SINGLE_MUTATIONS = {
    parent_idx: [
        (child_idx, nt_pos, new_base)
        for child_idx, nt_pos, new_base in mutations
        if child_idx not in STOP_CODON_IDXS  # Exclude stop codons
    ]
    for parent_idx, mutations in CODON_SINGLE_MUTATIONS.items()
    if parent_idx not in STOP_CODON_IDXS  # Exclude stop codons
}


def encode_codon_mutations(
    nt_parents: pd.Series, nt_children: pd.Series
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert parent/child nucleotide sequences to codon indices and mutation
    indicators.

    Args:
        nt_parents: Parent nucleotide sequences
        nt_children: Child nucleotide sequences

    Returns:
        Tuple of:
        - codon_parents_idxss: (N, L_codon) tensor of parent codon indices
        - codon_children_idxss: (N, L_codon) tensor of child codon indices
        - codon_mutation_indicators: (N, L_codon) boolean tensor indicating mutation positions

    Example:
        >>> parents = pd.Series(['ATGAAACCC'])
        >>> children = pd.Series(['ATGAAACCG'])  # CCC->CCG mutation
        >>> p_idx, c_idx, mut = encode_codon_mutations(parents, children)
        >>> mut[0]  # tensor([False, False, True])
    """
    # Convert sequences to lists for processing
    parent_seqs = nt_parents.tolist()
    child_seqs = nt_children.tolist()

    # Check that all sequences have same length and are multiples of 3
    if not all(len(seq) == len(parent_seqs[0]) for seq in parent_seqs + child_seqs):
        raise ValueError("All sequences must have the same length")

    seq_len = len(parent_seqs[0])
    if seq_len % 3 != 0:
        raise ValueError("Sequence length must be a multiple of 3")

    codon_len = seq_len // 3
    n_sequences = len(parent_seqs)

    # Extract all codons at once for vectorized processing
    all_parent_codons = []
    all_child_codons = []

    for parent_seq, child_seq in zip(parent_seqs, child_seqs):
        parent_codons = list(iter_codons(parent_seq))
        child_codons = list(iter_codons(child_seq))
        all_parent_codons.append(parent_codons)
        all_child_codons.append(child_codons)

    # Vectorized codon index lookup
    parent_codon_indices = torch.zeros((n_sequences, codon_len), dtype=torch.long)
    child_codon_indices = torch.zeros((n_sequences, codon_len), dtype=torch.long)
    mutation_indicators = torch.zeros((n_sequences, codon_len), dtype=torch.bool)

    # Process in batches for better cache locality
    for seq_idx in range(n_sequences):
        parent_codons = all_parent_codons[seq_idx]
        child_codons = all_child_codons[seq_idx]

        # Vectorized index lookup using list comprehension (faster than nested loops)
        parent_indices = [
            idx_of_codon_allowing_ambiguous(codon) for codon in parent_codons
        ]
        child_indices = [
            idx_of_codon_allowing_ambiguous(codon) for codon in child_codons
        ]
        mutations = [
            p_codon != c_codon for p_codon, c_codon in zip(parent_codons, child_codons)
        ]

        # Assign to tensors
        parent_codon_indices[seq_idx] = torch.tensor(parent_indices, dtype=torch.long)
        child_codon_indices[seq_idx] = torch.tensor(child_indices, dtype=torch.long)
        mutation_indicators[seq_idx] = torch.tensor(mutations, dtype=torch.bool)

    return parent_codon_indices, child_codon_indices, mutation_indicators


def create_codon_masks(nt_parents: pd.Series, nt_children: pd.Series) -> torch.Tensor:
    """Create masks for valid codon positions, masking ambiguous codons (containing Ns).

    Args:
        nt_parents: Parent nucleotide sequences
        nt_children: Child nucleotide sequences

    Returns:
        masks: (N, L_codon) boolean tensor indicating valid codon positions

    Example:
        >>> parents = pd.Series(['ATGNNNCCG'])  # Middle codon has Ns
        >>> children = pd.Series(['ATGNNNCCG'])
        >>> masks = create_codon_masks(parents, children)
        >>> masks[0]  # tensor([True, False, True])

    Raises:
        ValueError: If any sequences contain stop codons
    """
    # Convert sequences to lists for processing
    parent_seqs = nt_parents.tolist()
    child_seqs = nt_children.tolist()

    # Check for stop codons in all sequences
    for seq_idx, seq in enumerate(parent_seqs):
        if contains_stop_codon(seq):
            raise ValueError(f"Parent sequence {seq_idx} contains a stop codon: {seq}")

    for seq_idx, seq in enumerate(child_seqs):
        if contains_stop_codon(seq):
            raise ValueError(f"Child sequence {seq_idx} contains a stop codon: {seq}")

    # Check that all sequences have same length and are multiples of 3
    if not all(len(seq) == len(parent_seqs[0]) for seq in parent_seqs + child_seqs):
        raise ValueError("All sequences must have the same length")

    seq_len = len(parent_seqs[0])
    if seq_len % 3 != 0:
        raise ValueError("Sequence length must be a multiple of 3")

    codon_len = seq_len // 3
    n_sequences = len(parent_seqs)

    # Initialize mask tensor (True = valid, False = masked)
    masks = torch.ones((n_sequences, codon_len), dtype=torch.bool)

    # Process each sequence to identify ambiguous codons
    for seq_idx, (parent_seq, child_seq) in enumerate(zip(parent_seqs, child_seqs)):
        parent_codons = list(iter_codons(parent_seq))
        child_codons = list(iter_codons(child_seq))

        for codon_idx, (parent_codon, child_codon) in enumerate(
            zip(parent_codons, child_codons)
        ):
            # Mask positions where either parent or child has ambiguous codon (containing N)
            if "N" in parent_codon or "N" in child_codon:
                masks[seq_idx, codon_idx] = False

    return masks
