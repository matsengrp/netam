import pytest
import pandas as pd
import numpy as np
import torch
from Bio.Seq import Seq
from Bio.Data import CodonTable
from netam.sequences import (
    RESERVED_TOKENS,
    AA_STR_SORTED,
    TOKEN_REGEX,
    AA_TOKEN_STR_SORTED,
    CODONS,
    CODON_AA_INDICATOR_MATRIX,
    aa_onehot_tensor_of_str,
    nt_idx_array_of_str,
    nt_subs_indicator_tensor_of,
    translate_sequences,
    token_mask_of_aa_idxs,
    aa_idx_tensor_of_str,
)


def test_token_order():
    # If we always add additional tokens to the end, then converting to indices
    # will not be affected when we have a proper aa string.
    assert AA_TOKEN_STR_SORTED[: len(AA_STR_SORTED)] == AA_STR_SORTED


def test_token_replace():
    df = pd.DataFrame({"seq": ["AGCGTC" + token for token in AA_TOKEN_STR_SORTED]})
    newseqs = df["seq"].str.replace(TOKEN_REGEX, "N", regex=True)
    for seq, nseq in zip(df["seq"], newseqs):
        for token in RESERVED_TOKENS:
            seq = seq.replace(token, "N")
        assert nseq == seq


def test_token_mask():
    sample_aa_seq = "QYX^QC"
    mask = token_mask_of_aa_idxs(aa_idx_tensor_of_str(sample_aa_seq))
    for aa, mval in zip(sample_aa_seq, mask):
        if aa in RESERVED_TOKENS:
            assert mval
        else:
            assert not mval


def test_nucleotide_indices_of_codon():
    assert nt_idx_array_of_str("AAA").tolist() == [0, 0, 0]
    assert nt_idx_array_of_str("TAC").tolist() == [3, 0, 1]
    assert nt_idx_array_of_str("GCG").tolist() == [2, 1, 2]


def test_aa_onehot_tensor_of_str():
    aa_str = "QY"

    expected_output = torch.zeros((2, 20))
    expected_output[0][AA_STR_SORTED.index("Q")] = 1
    expected_output[1][AA_STR_SORTED.index("Y")] = 1

    output = aa_onehot_tensor_of_str(aa_str)

    assert output.shape == (2, 20)
    assert torch.equal(output, expected_output)


def test_translate_sequences():
    # sequence without stop codon
    seq_no_stop = ["AGTGGTGGTGGTGGTGGT"]
    assert translate_sequences(seq_no_stop) == [str(Seq(seq_no_stop[0]).translate())]

    # sequence with stop codon
    seq_with_stop = ["TAAGGTGGTGGTGGTAGT"]
    with pytest.raises(ValueError):
        translate_sequences(seq_with_stop)


def test_indicator_matrix():
    reconstructed_codon_table = {}
    indicator_matrix = CODON_AA_INDICATOR_MATRIX.numpy()

    for i, codon in enumerate(CODONS):
        row = indicator_matrix[i]
        if np.any(row):
            amino_acid = AA_STR_SORTED[np.argmax(row)]
            reconstructed_codon_table[codon] = amino_acid

    table = CodonTable.unambiguous_dna_by_id[1]  # 1 is for the standard table

    assert reconstructed_codon_table == table.forward_table


def test_subs_indicator_tensor_of():
    parent = "NAAA"
    child = "CAGA"
    expected_output = torch.tensor([0, 0, 1, 0], dtype=torch.float)
    output = nt_subs_indicator_tensor_of(parent, child)
    assert torch.equal(output, expected_output)
