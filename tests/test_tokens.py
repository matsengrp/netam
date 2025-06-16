import torch

from netam.sequences import (
    AA_AMBIG_IDX,
    MAX_KNOWN_TOKEN_COUNT,
    prepare_heavy_light_pair,
    translate_sequence,
    aa_idx_tensor_of_str,
    token_mask_of_aa_idxs,
    nt_mask_tensor_of,
    aa_mask_tensor_of,
    codon_mask_tensor_of,
    aa_strs_from_idx_tensor,
)


def test_mask_tensor_of():
    input_seq = "NAAA"
    # First test as nucleotides.
    expected_output = torch.tensor([0, 1, 1, 1, 0], dtype=torch.bool)
    output = nt_mask_tensor_of(input_seq, length=5)
    assert torch.equal(output, expected_output)
    # Next test as amino acids, where N counts as an AA.
    expected_output = torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)
    output = aa_mask_tensor_of(input_seq, length=5)
    assert torch.equal(output, expected_output)


def test_codon_mask_tensor_of():
    input_seq = "NAAAAAAAAAA"
    # First test as nucleotides.
    expected_output = torch.tensor([0, 1, 1, 0, 0], dtype=torch.bool)
    output = codon_mask_tensor_of(input_seq, aa_length=5)
    assert torch.equal(output, expected_output)
    input_seq2 = "AAAANAAAAAA"
    expected_output = torch.tensor([0, 0, 1, 0, 0], dtype=torch.bool)
    output = codon_mask_tensor_of(input_seq, input_seq2, aa_length=5)
    assert torch.equal(output, expected_output)


def test_aa_strs_from_idx_tensor():
    aa_idx_tensor = torch.tensor([[0, 1, 2, 3, 20, 21], [4, 5, 19, 21, 20, 20]])
    aa_strings = aa_strs_from_idx_tensor(aa_idx_tensor)
    assert aa_strings == ["ACDEX^", "FGY^"]


def test_mask_functions_agree(pcp_df, pcp_df_paired):
    for input_pcp_df in (pcp_df, pcp_df_paired):
        first_row = next(input_pcp_df.itertuples())
        seq = (first_row.parent_heavy, first_row.parent_light)

        for token_count in range(AA_AMBIG_IDX + 1, MAX_KNOWN_TOKEN_COUNT + 1):
            nt_seq_with_tokens = prepare_heavy_light_pair(
                *seq, MAX_KNOWN_TOKEN_COUNT, is_nt=True
            )[0]
            aa_seq_with_tokens = translate_sequence(nt_seq_with_tokens)
            aa_idx_seq = aa_idx_tensor_of_str(aa_seq_with_tokens)

            # We expect these to be the same because we use the first when
            # evaluating a model on a given amino acid sequence, and the second
            # when evaluating the model during loss computation in training.
            assert torch.allclose(
                aa_mask_tensor_of(aa_seq_with_tokens),
                codon_mask_tensor_of(nt_seq_with_tokens)
                | token_mask_of_aa_idxs(aa_idx_seq),
            )
