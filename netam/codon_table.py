import numpy as np
import torch

from Bio.Data import CodonTable
from netam.sequences import AA_STR_SORTED, CODONS, STOP_CODONS, translate_sequences


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
