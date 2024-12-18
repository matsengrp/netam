"""Code for handling sequences and sequence files."""

import itertools

import torch
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq

BASES = ["A", "C", "G", "T"]
BASES_AND_N_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
AA_STR_SORTED = "ACDEFGHIKLMNPQRSTVWY"
# ambiguous must remain last
TOKEN_STR_SORTED = AA_STR_SORTED + "^X"
RESERVED_TOKENS = ("^", )

RESERVED_TOKEN_AA_BOUNDS = (min(TOKEN_STR_SORTED.index(token) for token in RESERVED_TOKENS), max(TOKEN_STR_SORTED.index(token) for token in RESERVED_TOKENS))

NT_STR_SORTED = "ACGT"
MAX_AA_TOKEN_IDX = len(TOKEN_STR_SORTED) - 1
CODONS = [
    "".join(codon_list)
    for codon_list in itertools.product(["A", "C", "G", "T"], repeat=3)
]
STOP_CODONS = ["TAA", "TAG", "TGA"]
TOKEN_TRANSLATIONS = {
    "^^^": "^",
}


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
        return np.array([TOKEN_STR_SORTED.index(aa) for aa in aa_str])
    except ValueError:
        print(f"Found an invalid amino acid in the string: {aa_str}")
        raise

def aa_idx_array_of_str(aa_str):
    """Return the indices of the amino acids in a string."""
    try:
        return np.array([TOKEN_STR_SORTED.index(aa) for aa in aa_str])
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
        return torch.tensor([TOKEN_STR_SORTED.index(aa) for aa in aa_str])
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


def translate_codon(codon):
    """Translate a codon to an amino acid."""
    if codon in TOKEN_TRANSLATIONS:
        return TOKEN_TRANSLATIONS[codon]
    else:
        return str(Seq(codon).translate())


def translate_sequence(nt_sequence):
    if len(nt_sequence) % 3 != 0:
        raise ValueError(f"The sequence '{nt_sequence}' is not a multiple of 3.")
    aa_seq = "".join(translate_codon(nt_sequence[i: i + 3]) for i in range(0, len(nt_sequence), 3))
    if "*" in aa_seq:
        raise ValueError(f"The sequence '{nt_sequence}' contains a stop codon.")
    return aa_seq


def translate_sequences(nt_sequences):
    return [translate_sequence(seq) for seq in nt_sequences]


def aa_index_of_codon(codon):
    """Return the index of the amino acid encoded by a codon."""
    aa = translate_sequence(codon)
    return TOKEN_STR_SORTED.index(aa)


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

    matrix = np.zeros((len(CODONS), len(TOKEN_STR_SORTED)))

    for i, codon in enumerate(CODONS):
        try:
            aa = translate_sequences([codon])[0]
            aa_idx = TOKEN_STR_SORTED.index(aa)
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


def apply_aa_mask_to_nt_sequence(nt_seq, aa_mask):
    """Apply an amino acid mask to a nucleotide sequence."""
    return "".join(
        nt for nt, mask_val in zip(nt_seq, aa_mask.repeat_interleave(3)) if mask_val
    )


def iter_codons(nt_seq):
    """Iterate over the codons in a nucleotide sequence."""
    for i in range(0, (len(nt_seq) // 3) * 3, 3):
        yield nt_seq[i : i + 3]


# def ambig_mask_of_nt_string(nt_str):
#     """Return a mask indicating which positions in a nucleotide sequence are not N."""
#     return torch.tensor([nt != "N" for nt in nt_str], dtype=torch.bool)


def token_mask_of_aa_idxs(aa_idxs: torch.Tensor) -> torch.Tensor:
    """Return a mask indicating which positions in an amino acid sequence contain special indicator tokens"""
    min_idx, max_idx = RESERVED_TOKEN_AA_BOUNDS
    return (aa_idxs <= max_idx) & (aa_idxs >= min_idx)
