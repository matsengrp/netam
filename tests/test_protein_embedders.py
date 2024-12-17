from netam import protein_embedders


def test_tokenize_sequences():
    embedder = protein_embedders.ESMEmbedder(model_name="esm2_t6_8M_UR50D")
    sequences = [
        "EAQLVESGGGLVQPGGSLRLSCAASGFTISDYWIHWVRQAPGKGLEW",
        "ECQLVESGGGLVQPGGSLRLSCAASGFTISDYWIHWVRQAPGKGLEW",
    ]
    tokens = embedder.tokenize_sequences(sequences)
    assert tokens.shape == (2, 49)
    assert tokens[0][0] == embedder.alphabet.cls_idx
    assert tokens[0][-1] == embedder.alphabet.eos_idx
