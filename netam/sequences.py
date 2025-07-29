"""Code for handling sequences and sequence files."""

import itertools

import torch
import re
import math
import pandas as pd
import numpy as np

from torch import nn, Tensor

from Bio import SeqIO
from Bio.Seq import Seq

from netam.common import combine_and_pad_tensors

BASES = ("A", "C", "G", "T")
AA_STR_SORTED = "ACDEFGHIKLMNPQRSTVWY"


NT_STR_SORTED = "".join(BASES)
BASES_AND_N_TO_INDEX = {base: idx for idx, base in enumerate(NT_STR_SORTED + "N")}
AA_AMBIG_IDX = len(AA_STR_SORTED)
# Used for padding amino acid sequences to the same length. Differentiated by
# name in case we add a padding token other than AA_AMBIG_IDX later.
AA_PADDING_TOKEN = AA_AMBIG_IDX

CODONS = ["".join(codon_list) for codon_list in itertools.product(BASES, repeat=3)]
STOP_CODONS = ["TAA", "TAG", "TGA"]
AMBIGUOUS_CODON_IDX = len(CODONS)

# Create a dictionary for O(1) codon index lookups
CODON_TO_INDEX = {codon: idx for idx, codon in enumerate(CODONS)}


# Add additional tokens to this string:
RESERVED_TOKENS = "^"
# Each token in RESERVED_TOKENS will appear once in aa strings, and three times
# in nt strings.
RESERVED_TOKEN_TRANSLATIONS = {token * 3: token for token in RESERVED_TOKENS}

# Must add new tokens to the end of this string.
TOKEN_STR_SORTED = AA_STR_SORTED + "X" + RESERVED_TOKENS

RESERVED_TOKEN_AA_BOUNDS = (
    min(TOKEN_STR_SORTED.index(token) for token in RESERVED_TOKENS),
    max(TOKEN_STR_SORTED.index(token) for token in RESERVED_TOKENS),
)
MAX_KNOWN_TOKEN_COUNT = len(TOKEN_STR_SORTED)
MAX_AA_TOKEN_IDX = MAX_KNOWN_TOKEN_COUNT - 1


# Create a regex pattern
RESERVED_TOKEN_REGEX = f"[{''.join(map(re.escape, list(RESERVED_TOKENS)))}]"
HL_SEPARATOR_TOKEN_IDX = TOKEN_STR_SORTED.index("^")

# I needed some sequence to use to normalize the rate of mutation in the SHM model.
# So, I chose perhaps the most famous antibody sequence, VRC01:
# https://www.ncbi.nlm.nih.gov/nuccore/GU980702.1
VRC01_NT_SEQ = (
    "CAGGTGCAGCTGGTGCAGTCTGGGGGTCAGATGAAGAAGCCTGGCGAGTCGATGAGAATT"
    "TCTTGTCGGGCTTCTGGATATGAATTTATTGATTGTACGCTAAATTGGATTCGTCTGGCC"
    "CCCGGAAAAAGGCCTGAGTGGATGGGATGGCTGAAGCCTCGGGGGGGGGCCGTCAACTAC"
    "GCACGTCCACTTCAGGGCAGAGTGACCATGACTCGAGACGTTTATTCCGACACAGCCTTT"
    "TTGGAGCTGCGCTCGTTGACAGTAGACGACACGGCCGTCTACTTTTGTACTAGGGGAAAA"
    "AACTGTGATTACAATTGGGACTTCGAACACTGGGGCCGGGGCACCCCGGTCATCGTCTCA"
    "TCA"
)

CODON_TO_NT_TENSOR = torch.arange(64).reshape(4, 4, 4)

# Create the reverse mapping: from codon indices to nt indices
NT_TO_CODON_TENSOR = torch.zeros((64, 3), dtype=torch.long)
for i in range(4):
    for j in range(4):
        for k in range(4):
            codon_idx = CODON_TO_NT_TENSOR[i, j, k]
            NT_TO_CODON_TENSOR[codon_idx, 0] = i
            NT_TO_CODON_TENSOR[codon_idx, 1] = j
            NT_TO_CODON_TENSOR[codon_idx, 2] = k


def unflatten_codon_idxs(codon_idx_tensor):
    """Convert tensor of codon indices to tensor of shape (L, 3) containing nucleotide
    indices.

    Args:
        codon_idx_tensor: Tensor of shape (L,) containing codon indices (0-63)

    Returns:
        Tensor of shape (L, 3) containing the corresponding nucleotide indices
    """
    return NT_TO_CODON_TENSOR[codon_idx_tensor]


def flatten_codon_idxs(nt_codon_tensor):
    """Convert codon index tensor of shape (L, 3) containing nucleotide indices to
    tensor of length L containing codon indices."""
    original_shape = nt_codon_tensor.shape
    flat_codon_tensor = nt_codon_tensor.view(-1, 3)
    return CODON_TO_NT_TENSOR[
        flat_codon_tensor[:, 0], flat_codon_tensor[:, 1], flat_codon_tensor[:, 2]
    ].reshape(*original_shape[:-1])


def idx_of_codon_allowing_ambiguous(codon):
    if "N" in codon:
        return AMBIGUOUS_CODON_IDX
    else:
        return CODON_TO_INDEX[codon]


def codon_idx_tensor_of_str_ambig(nt_str):
    """Return the indices of the codons in a string."""
    assert len(nt_str) % 3 == 0
    return torch.tensor(
        [idx_of_codon_allowing_ambiguous(codon) for codon in iter_codons(nt_str)]
    )


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
    if codon in RESERVED_TOKEN_TRANSLATIONS:
        return RESERVED_TOKEN_TRANSLATIONS[codon]
    else:
        return str(Seq(codon).translate())


def translate_sequence(nt_sequence):
    if len(nt_sequence) % 3 != 0:
        raise ValueError(f"The sequence '{nt_sequence}' is not a multiple of 3.")
    aa_seq = "".join(
        translate_codon(nt_sequence[i : i + 3]) for i in range(0, len(nt_sequence), 3)
    )
    if "*" in aa_seq:
        raise ValueError(f"The sequence '{nt_sequence}' contains a stop codon.")
    return aa_seq


def translate_sequences(nt_sequences):
    return [translate_sequence(seq) for seq in nt_sequences]


def translate_sequence_mask_codons(nt_sequence):
    """Translate a nucleotide sequence, masking as ambiguous any codon containing an
    N."""
    return "".join(
        translate_codon(codon) if "N" not in codon else "X"
        for codon in iter_codons(nt_sequence)
    )


def translate_sequences_mask_codons(nt_sequences):
    return [translate_sequence_mask_codons(seq) for seq in nt_sequences]


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


def contains_stop_codon(nt_seq):
    """Check if a nucleotide sequence contains a stop codon."""
    return any(codon in STOP_CODONS for codon in iter_codons(nt_seq))


def prepare_heavy_light_pair(heavy_seq, light_seq, known_token_count, is_nt=True):
    """Prepare a pair of heavy and light chain sequences for model input.

    Args:
        heavy_seq (str): The heavy chain sequence.
        light_seq (str): The light chain sequence.
        known_token_count (int): The number of tokens recognized by the model which will take the result as input.
        is_nt (bool): Whether the sequences are nucleotide sequences. Otherwise, they
            are assumed to be amino acid sequences.
    Returns:
        The prepared sequence, and a tuple of indices indicating positions where tokens were added to the prepared sequence.
    """
    # In the future, we'll define a list of functions that will be applied in
    # order, up to the maximum number of accepted tokens.
    if known_token_count > AA_AMBIG_IDX + 1:
        if is_nt:
            heavy_light_separator = "^^^"
        else:
            heavy_light_separator = "^"

        prepared_seq = heavy_seq + heavy_light_separator + light_seq
        added_indices = tuple(
            range(len(heavy_seq), len(heavy_seq) + len(heavy_light_separator))
        )
    else:
        prepared_seq = heavy_seq
        added_indices = tuple()

    return prepared_seq, added_indices


def heavy_light_mask_of_aa_idxs(aa_idxs):
    """Return a mask indicating which positions in a single amino acid sequence are in
    the heavy chain, and which positions are in the light chain. The returned value is a
    dictionary with keys `heavy` and `light`, and torch mask tensors as values.

    The returned masks are True only for actual amino acid positions, never for reserved
    tokens.
    """
    # As written, can only handle single sequences.
    assert len(aa_idxs.shape) == 1
    is_not_token = ~token_mask_of_aa_idxs(aa_idxs)
    separator_indices = torch.where(aa_idxs == HL_SEPARATOR_TOKEN_IDX)[0]
    if len(separator_indices) < 1:
        # assume all heavy chain
        return {
            "heavy": is_not_token,
            "light": torch.full_like(aa_idxs, False, dtype=torch.bool),
        }
    elif len(separator_indices) == 1:
        before_separator = torch.arange(len(aa_idxs)) < separator_indices[0]
        after_separator = torch.arange(len(aa_idxs)) > separator_indices[0]
        return {
            "heavy": is_not_token & before_separator,
            "light": is_not_token & after_separator,
        }
    else:
        raise ValueError(
            f"Expected exactly zero or one separator tokens in the sequence, found {len(separator_indices)}."
        )

    return aa_idxs < AA_AMBIG_IDX


def split_heavy_light_model_outputs(result, aa_idxs):
    """Split a tensor whose first dimension corresponds to amino acid positions into
    heavy chain and light chain components.

    Args:
        result: The tensor to split.
        aa_idxs: The amino acid indices corresponding to the tensor, as presented to the model (including any special tokens).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The heavy chain and light chain components of the input tensor.
    """
    # Now split into heavy and light chain results:
    heavy_mask, light_mask = heavy_light_mask_of_aa_idxs(aa_idxs).values()
    heavy_chain = result[heavy_mask]
    light_chain = result[light_mask]
    return heavy_chain, light_chain


def dataset_inputs_of_pcp_df(pcp_df, known_token_count):
    parents = []
    children = []
    nt_ratess = []
    nt_cspss = []
    for row in pcp_df.itertuples():
        parent, parent_token_idxs = prepare_heavy_light_pair(
            row.parent_heavy, row.parent_light, known_token_count, is_nt=True
        )
        child = prepare_heavy_light_pair(
            row.child_heavy, row.child_light, known_token_count, is_nt=True
        )[0]
        # These are the fill values that the neutral model returns when given N's:
        nt_rates = combine_and_pad_tensors(
            row.nt_rates_heavy, row.nt_rates_light, parent_token_idxs, fill=1.0
        )[: len(parent)]
        nt_csps = combine_and_pad_tensors(
            row.nt_csps_heavy, row.nt_csps_light, parent_token_idxs, fill=0.0
        )[: len(parent)]
        parents.append(parent)
        children.append(child)
        nt_ratess.append(nt_rates)
        nt_cspss.append(nt_csps)

    return tuple(
        map(
            pd.Series,
            (
                parents,
                children,
                nt_ratess,
                nt_cspss,
            ),
        )
    )


def generate_kmers(kmer_length):
    # Our strategy for kmers is to have a single representation for any kmer that isn't in ACGT.
    # This is the first one, which is simply "N", and so this placeholder value is 0.
    all_kmers = ["N"] + [
        "".join(p) for p in itertools.product(BASES, repeat=kmer_length)
    ]
    assert len(all_kmers) < torch.iinfo(torch.int32).max
    return all_kmers


def kmer_to_index_of(all_kmers):
    return {kmer: idx for idx, kmer in enumerate(all_kmers)}


def aa_idx_tensor_of_str_ambig(aa_str):
    """Return the indices of the amino acids in a string, allowing the ambiguous
    character."""
    try:
        return torch.tensor(
            [TOKEN_STR_SORTED.index(aa) for aa in aa_str], dtype=torch.int
        )
    except ValueError:
        print(f"Found an invalid amino acid in the string: {aa_str}")
        raise


def generic_mask_tensor_of(ambig_symb, seq_str, length=None):
    """Return a mask tensor indicating non-empty and non-ambiguous sites.

    Sites beyond the length of the sequence are masked.
    """
    if length is None:
        length = len(seq_str)
    mask = torch.zeros(length, dtype=torch.bool)
    if len(seq_str) < length:
        seq_str += ambig_symb * (length - len(seq_str))
    else:
        seq_str = seq_str[:length]
    mask[[c != ambig_symb for c in seq_str]] = 1
    return mask


def aa_strs_from_idx_tensor(idx_tensor):
    """Convert a tensor of amino acid indices back to a list of amino acid strings.

    Args:
        idx_tensor (Tensor): A 2D tensor of shape (batch_size, seq_len) containing
                             indices into TOKEN_STR_SORTED.

    Returns:
        List[str]: A list of amino acid strings with trailing 'X's removed.
    """
    idx_tensor = idx_tensor.cpu()

    aa_str_list = []
    for row in idx_tensor:
        aa_str = "".join(TOKEN_STR_SORTED[idx] for idx in row.tolist())
        aa_str_list.append(aa_str.rstrip("X"))

    return aa_str_list


def nt_mask_tensor_of(*args, **kwargs):
    return generic_mask_tensor_of("N", *args, **kwargs)


def aa_mask_tensor_of(*args, **kwargs):
    return generic_mask_tensor_of("X", *args, **kwargs)


def _consider_codon(codon):
    """Return False if codon should be masked, True otherwise."""
    if "N" in codon:
        return False
    elif codon in RESERVED_TOKEN_TRANSLATIONS:
        return False
    else:
        return True


def codon_mask_tensor_of(nt_parent, *other_nt_seqs, aa_length=None):
    """Return a mask tensor indicating codons which contain at least one N.

    Codons beyond the length of the sequence are masked. If other_nt_seqs are provided,
    the "and" mask will be computed for all sequences. Codons containing marker tokens
    are also masked.
    """
    if aa_length is None:
        aa_length = len(nt_parent) // 3
    sequences = (nt_parent,) + other_nt_seqs
    mask = [
        all(_consider_codon(codon) for codon in codons)
        for codons in zip(*(iter_codons(sequence) for sequence in sequences))
    ]
    if len(mask) < aa_length:
        mask += [False] * (aa_length - len(mask))
    else:
        mask = mask[:aa_length]
    assert len(mask) == aa_length
    return torch.tensor(mask, dtype=torch.bool)


def is_pcp_valid(parent, child, aa_mask=None):
    """Check that the parent-child pairs are valid. Returns True if valid, False
    otherwise. To be valid, the following conditions must be met:

    * The parent and child sequences must be the same length
    * There must be unmasked codons
    * The parent and child sequences must not match after masking codons containing
      ambiguities.

    Args:
        parent: The parent sequence.
        child: The child sequence.
        aa_mask: The mask tensor for the amino acid sequence. If None, it will be
            computed from the parent and child sequences.
    """
    # Note that the use of try/except within control flow is generally not
    # recommended, but in this case it allows for more granuality in the
    # error messages.
    try:
        assert_pcp_valid(parent, child, aa_mask)
        return True
    except ValueError:
        return False


def assert_pcp_valid(parent, child, aa_mask=None):
    """Check that the parent-child pairs are valid. Raises a ValueError if not. To be valid,
    the following conditions must be met:

    * The parent and child sequences must be the same length
    * There must be unmasked codons
    * The parent and child sequences must not match after masking codons containing
      ambiguities.

    Args:
        parent: The parent sequence.
        child: The child sequence.
        aa_mask: The mask tensor for the amino acid sequence. If None, it will be
            computed from the parent and child sequences.
    """
    if aa_mask is None:
        aa_mask = codon_mask_tensor_of(parent, child)
    if len(parent) != len(child):
        raise ValueError("The parent and child sequences are not the same length.")
    if not aa_mask.any():
        raise ValueError("The parent and child sequences are masked in all codons.")
    if apply_aa_mask_to_nt_sequence(parent, aa_mask) == apply_aa_mask_to_nt_sequence(
        child, aa_mask
    ):
        raise ValueError(
            "The parent and child nucleotide sequence "
            "pair matches after masking codons containing ambiguities. "
            "To avoid this try filtering data using `netam.sequences.assert_pcp_valid`."
        )


# Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # assert that d_model is even
        assert d_model % 2 == 0, "d_model must be even for PositionalEncoding"

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def encode_sequences(sequences, encoder):
    encoded_parents, wt_base_modifiers = zip(
        *[encoder.encode_sequence(sequence) for sequence in sequences]
    )
    masks = [nt_mask_tensor_of(sequence, encoder.site_count) for sequence in sequences]
    return (
        torch.stack(encoded_parents),
        torch.stack(masks),
        torch.stack(wt_base_modifiers),
    )


def token_mask_of_aa_idxs(aa_idxs: torch.Tensor) -> torch.Tensor:
    """Return a mask indicating which positions in an amino acid sequence contain
    special indicator tokens.

    The mask is True for positions that contain special tokens.
    """
    min_idx, max_idx = RESERVED_TOKEN_AA_BOUNDS
    return (aa_idxs <= max_idx) & (aa_idxs >= min_idx)


def aa_index_of_codon(codon):
    """Return the index of the amino acid encoded by a codon."""
    aa = translate_sequence(codon)
    return TOKEN_STR_SORTED.index(aa)


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


def hamming_distance(seq1, seq2):
    """Calculate the Hamming distance between two sequences."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length.")
    return sum(el1 != el2 for el1, el2 in zip(seq1, seq2))


def paired_hamming_distance(seq1, seq2):
    return sum(hamming_distance(s1, s2) for s1, s2 in zip(seq1, seq2))
