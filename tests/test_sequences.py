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
    MAX_KNOWN_TOKEN_COUNT,
    AA_AMBIG_IDX,
    AMBIGUOUS_CODON_IDX,
    aa_onehot_tensor_of_str,
    codon_idx_tensor_of_str_ambig,
    nt_idx_array_of_str,
    nt_subs_indicator_tensor_of,
    translate_sequences,
    token_mask_of_aa_idxs,
    aa_idx_tensor_of_str,
    prepare_heavy_light_pair,
    dataset_inputs_of_pcp_df,
    heavy_light_mask_of_aa_idxs,
    unflatten_codon_idxs,
    flatten_codon_idxs,
    nt_idx_tensor_of_str,
    codon_mask_tensor_of,
)
from netam.codon_table import CODON_AA_INDICATOR_MATRIX
from netam.common import combine_and_pad_tensors


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


def test_heavy_light_mask():
    test_pairs = [
        ("AGCGNCCCN", "AGCGTC"),
        ("AGCGNCCCT", ""),
        ("", "AGCGTC"),
    ]
    for TOKEN_COUNT in range(AA_AMBIG_IDX + 1, MAX_KNOWN_TOKEN_COUNT + 1):
        for heavy, light in test_pairs:
            heavy_aa_idxs = aa_idx_tensor_of_str(translate_sequences([heavy])[0])
            light_aa_idxs = aa_idx_tensor_of_str(translate_sequences([light])[0])
            prepared, _ = prepare_heavy_light_pair(heavy, light, TOKEN_COUNT)
            prepared_aa = translate_sequences([prepared])[0]
            prepared_idxs = aa_idx_tensor_of_str(prepared_aa)
            hlmasks = heavy_light_mask_of_aa_idxs(prepared_idxs)
            assert torch.allclose(
                prepared_idxs[hlmasks["heavy"]].long(), heavy_aa_idxs.long()
            )
            # The separator token is the next token after the ambiguous token.
            if TOKEN_COUNT > AA_AMBIG_IDX + 1:
                print(prepared_idxs[hlmasks["light"]], light_aa_idxs)
                assert torch.allclose(
                    prepared_idxs[hlmasks["light"]].long(), light_aa_idxs.long()
                )
            else:
                assert not hlmasks["light"].any()


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


def test_codon_idx_tensor_of_str():
    nt_str = "AAAAACTTGTTTNTT"
    expected_output = torch.tensor([0, 1, 62, 63, AMBIGUOUS_CODON_IDX])
    output = codon_idx_tensor_of_str_ambig(nt_str)
    assert torch.equal(output, expected_output)


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
                *dataset_inputs_of_pcp_df(df, token_count)
            ):
                assert len(nt_rates) == len(parent)
                assert len(nt_csps) == len(parent)
                assert len(parent) == len(child)

    # Here we just make sure for the largest possible token count, that csps
    # and rates are padded in the correct places.
    for df in (pcp_df, pcp_df_paired):
        for parent, child, nt_rates, nt_csps in zip(
            *dataset_inputs_of_pcp_df(df, MAX_KNOWN_TOKEN_COUNT)
        ):
            for idx in range(len(parent)):
                if parent[idx] in RESERVED_TOKENS:
                    assert torch.allclose(nt_rates[idx], torch.tensor(1.0))
                    assert torch.allclose(nt_csps[idx], torch.tensor(0.0))


def test_codon_indices():
    seq = "ATGCGTACGTAG"
    true_codon_indices = codon_idx_tensor_of_str_ambig(seq)
    true_nt_codon_indices = nt_idx_tensor_of_str(seq).view(-1, 3)

    test_codon_idxs = flatten_codon_idxs(true_nt_codon_indices)
    assert torch.allclose(test_codon_idxs, true_codon_indices)

    test_nt_codon_idxs = unflatten_codon_idxs(true_codon_indices)
    assert torch.allclose(test_nt_codon_idxs, true_nt_codon_indices)

    # Now see if they work with extra dimensions
    true_codon_indices = true_codon_indices.unsqueeze(0)
    true_nt_codon_indices = true_nt_codon_indices.unsqueeze(0)

    test_nt_codon_idxs = unflatten_codon_idxs(true_codon_indices)
    assert torch.allclose(test_nt_codon_idxs, true_nt_codon_indices)

    test_codon_idxs = flatten_codon_idxs(true_nt_codon_indices)

    print(test_codon_idxs)
    print(true_codon_indices)
    if not torch.allclose(test_codon_idxs, true_codon_indices):
        print(test_codon_idxs)
        print(true_codon_indices)
        assert False


def test_codon_mask_tensor_of():
    cases = [
        [
            "ATGCGTACGTAG",
            None,
            [True] * 4,
        ],
        [
            "ATGCGTACGTAG",
            4,
            [True] * 4,
        ],
        [
            "ATGCGTACGTAG",
            5,
            [True] * 4 + [False],
        ],
        [
            "ATGCGTACGTAG",
            3,
            [True] * 3,
        ],
        [
            "ATGCGTACGTAN",
            None,
            [True] * 3 + [False],
        ],
        [
            "ATG^^^ACGTAG",
            4,
            [True, False, True, True],
        ],
        [
            "ATGNGTACGTAG",
            5,
            [True, False, True, True, False],
        ],
        [
            "ATGCGTACGNAG",
            3,
            [True] * 3,
        ],
    ]
    for seq, aa_len, expected in cases:
        assert codon_mask_tensor_of(seq, aa_length=aa_len).tolist() == expected
