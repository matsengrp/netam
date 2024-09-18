"""Code for handling sequences and sequence files."""

import itertools

import torch
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq

AA_STR_SORTED = "ACDEFGHIKLMNPQRSTVWY"
NT_STR_SORTED = "ACGT"
CODONS = [
    "".join(codon_list)
    for codon_list in itertools.product(["A", "C", "G", "T"], repeat=3)
]
STOP_CODONS = ["TAA", "TAG", "TGA"]


def nt_idx_array_of_str(nt_str):
    """Return the indices of the nucleotides in a string."""
    try:
        return np.array([NT_STR_SORTED.index(nt) for nt in nt_str])
    except ValueError:
        print(f"Found an invalid nucleotide in the string: {nt_str}")
        raise


def aa_idx_array_of_str(aa_str):
    """Return the indices of the amino acids in a string."""
    try:
        return np.array([AA_STR_SORTED.index(aa) for aa in aa_str])
    except ValueError:
        print(f"Found an invalid amino acid in the string: {aa_str}")
        raise


def nt_idx_tensor_of_str(nt_str):
    """Return the indices of the nucleotides in a string."""
    try:
        return torch.tensor([NT_STR_SORTED.index(nt) for nt in nt_str])
    except ValueError:
        print(f"Found an invalid nucleotide in the string: {nt_str}")
        raise


def aa_idx_tensor_of_str(aa_str):
    """Return the indices of the amino acids in a string."""
    try:
        return torch.tensor([AA_STR_SORTED.index(aa) for aa in aa_str])
    except ValueError:
        print(f"Found an invalid amino acid in the string: {aa_str}")
        raise


def aa_onehot_tensor_of_str(aa_str):
    aa_onehot = torch.zeros((len(aa_str), 20))
    aa_indices_parent = aa_idx_array_of_str(aa_str)
    aa_onehot[torch.arange(len(aa_str)), aa_indices_parent] = 1
    return aa_onehot


def generic_subs_indicator_tensor_of(ambig_symb, parent, child):
    """Return a tensor indicating which positions in the parent sequence are substituted
    in the child sequence."""
    return torch.tensor(
        [
            0 if (p == ambig_symb or c == ambig_symb) else p != c
            for p, c in zip(parent, child)
        ],
        dtype=torch.float,
    )


def nt_subs_indicator_tensor_of(parent, child):
    """Return a tensor indicating which positions in the parent sequence are substituted
    in the child sequence."""
    return generic_subs_indicator_tensor_of("N", parent, child)


def aa_subs_indicator_tensor_of(parent, child):
    """Return a tensor indicating which positions in the parent sequence are substituted
    in the child sequence."""
    return generic_subs_indicator_tensor_of("X", parent, child)


def read_fasta_sequences(file_path):
    with open(file_path, "r") as handle:
        sequences = [str(record.seq) for record in SeqIO.parse(handle, "fasta")]
    return sequences


def translate_sequences(nt_sequences):
    aa_sequences = []
    for seq in nt_sequences:
        if len(seq) % 3 != 0:
            raise ValueError(f"The sequence '{seq}' is not a multiple of 3.")
        aa_seq = str(Seq(seq).translate())
        if "*" in aa_seq:
            raise ValueError(f"The sequence '{seq}' contains a stop codon.")
        aa_sequences.append(aa_seq)
    return aa_sequences


def translate_sequence(nt_sequence):
    return translate_sequences([nt_sequence])[0]


def aa_index_of_codon(codon):
    """Return the index of the amino acid encoded by a codon."""
    aa = translate_sequence(codon)
    return AA_STR_SORTED.index(aa)


def generic_mutation_frequency(ambig_symb, parent, child):
    """Return the fraction of sites that differ between the parent and child
    sequences."""
    return sum(
        1
        for p, c in zip(parent, child)
        if p != c and p != ambig_symb and c != ambig_symb
    ) / len(parent)


def nt_mutation_frequency(parent, child):
    """Return the fraction of nucleotide sites that differ between the parent and child
    sequences."""
    return generic_mutation_frequency("N", parent, child)


def aa_mutation_frequency(parent, child):
    """Return the fraction of amino acid sites that differ between the parent and child
    sequences."""
    return generic_mutation_frequency("X", parent, child)


def assert_pcp_lengths(parent, child):
    """Assert that the lengths of the parent and child sequences are the same and that
    they are multiples of 3."""
    if len(parent) != len(child):
        raise ValueError(
            f"The parent and child sequences are not the same length: "
            f"{len(parent)} != {len(child)}"
        )
    if len(parent) % 3 != 0:
        raise ValueError(f"Found a PCP with length not a multiple of 3: {len(parent)}")


def pcp_criteria_check(parent, child, max_mut_freq=0.3):
    """Check that parent child pair undergoes mutation at a reasonable rate."""
    if parent == child:
        return False
    elif nt_mutation_frequency(parent, child) > max_mut_freq:
        return False
    else:
        return True


def generate_codon_aa_indicator_matrix():
    """Generate a matrix that maps codons (rows) to amino acids (columns)."""

    matrix = np.zeros((len(CODONS), len(AA_STR_SORTED)))

    for i, codon in enumerate(CODONS):
        try:
            aa = translate_sequences([codon])[0]
            aa_idx = AA_STR_SORTED.index(aa)
            matrix[i, aa_idx] = 1
        except ValueError:  # Handle STOP codon
            pass

    return matrix


CODON_AA_INDICATOR_MATRIX = torch.tensor(
    generate_codon_aa_indicator_matrix(), dtype=torch.float32
)


def assert_full_sequences(parent, child):
    """Assert that the parent and child sequences full length, containing no ambiguous
    bases (N)."""

    if "N" in parent or "N" in child:
        raise ValueError("Found ambiguous bases in the parent or child sequence.")
