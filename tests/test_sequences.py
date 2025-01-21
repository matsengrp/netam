import pytest
import pandas as pd
import numpy as np
import torch
from Bio.Seq import Seq
from Bio.Data import CodonTable
from netam.sequences import (
    RESERVED_TOKENS,
    AA_STR_SORTED,
    RESERVED_TOKEN_REGEX,
    TOKEN_STR_SORTED,
    CODONS,
    CODON_AA_INDICATOR_MATRIX,
    MAX_KNOWN_TOKEN_COUNT,
    AA_AMBIG_IDX,
    aa_onehot_tensor_of_str,
    nt_idx_array_of_str,
    nt_subs_indicator_tensor_of,
    translate_sequences,
    token_mask_of_aa_idxs,
    aa_idx_tensor_of_str,
    prepare_heavy_light_pair,
    combine_and_pad_tensors,
    dataset_inputs_of_pcp_df,
)


def test_token_order():
    # If we always add additional tokens to the end, then converting to indices
    # will not be affected when we have a proper aa string.
    assert TOKEN_STR_SORTED[: len(AA_STR_SORTED)] == AA_STR_SORTED


def test_token_replace():
    df = pd.DataFrame({"seq": ["AGCGTC" + token for token in TOKEN_STR_SORTED]})
    newseqs = df["seq"].str.replace(RESERVED_TOKEN_REGEX, "N", regex=True)
    for seq, nseq in zip(df["seq"], newseqs):
        for token in RESERVED_TOKENS:
            seq = seq.replace(token, "N")
        assert nseq == seq


def test_prepare_heavy_light_pair():
    heavy = "AGCGTC"
    light = "AGCGTC"
    for heavy, light in [
        ("AGCGTC", "AGCGTC"),
        ("AGCGTC", ""),
        ("", "AGCGTC"),
    ]:
        assert prepare_heavy_light_pair(heavy, light, MAX_KNOWN_TOKEN_COUNT) == (
            heavy + "^^^" + light,
            tuple(range(len(heavy), len(heavy) + 3)),
        )

    heavy = "QVQ"
    light = "QVQ"
    for heavy, light in [
        ("QVQ", "QVQ"),
        ("QVQ", ""),
        ("", "QVQ"),
    ]:
        assert prepare_heavy_light_pair(
            heavy, light, MAX_KNOWN_TOKEN_COUNT, is_nt=False
        ) == (heavy + "^" + light, tuple(range(len(heavy), len(heavy) + 1)))


def test_combine_and_pad_tensors():
    # Test that function works with 1d tensors:
    t1 = torch.tensor([1, 2, 3], dtype=torch.float)
    t2 = torch.tensor([4, 5, 6], dtype=torch.float)
    idxs = (0, 4, 5)
    result = combine_and_pad_tensors(t1, t2, idxs)
    mask = result.isnan()
    assert torch.equal(
        result[~mask], torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)
    )
    assert all(mask[torch.tensor(idxs)])


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


def test_dataset_inputs_of_pcp_df(pcp_df, pcp_df_paired):
    for token_count in range(AA_AMBIG_IDX + 1, MAX_KNOWN_TOKEN_COUNT + 1):
        for df in (pcp_df, pcp_df_paired):
            for parent, child, nt_rates, nt_csps in zip(
                *dataset_inputs_of_pcp_df(df, 22)
            ):
                assert len(nt_rates) == len(parent)
                assert len(nt_csps) == len(parent)
                assert len(parent) == len(child)
