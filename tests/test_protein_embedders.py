import pytest

from netam import protein_embedders


@pytest.fixture
def embedder():
    return protein_embedders.ESMEmbedder(model_name="esm2_t6_8M_UR50D")


@pytest.fixture
def sequences():
    return [
        "EAQLVESGGGLVQPGGSLRLSCAASGFTISDYWIHWVRQAPGKGLEW",
        "ECQLVESGGGLVQPGGSLRLSCAASGFTISDYWIHWVRQAPGKGLEW",
    ]


def test_tokenize_sequences(embedder, sequences):
    tokens = embedder.tokenize_sequences(sequences)
    assert tokens.shape == (2, 49)
    assert tokens[0][0] == embedder.alphabet.cls_idx
    assert tokens[0][-1] == embedder.alphabet.eos_idx


def test_roundtrip(embedder, sequences):
    """Embed sequences, and de-embed them back to sequences.

    Note that we don't get the exact same sequences back! We just get back mostly the
    same amino acids.
    """
    tokens = embedder.tokenize_sequences(sequences)
    embeddings = embedder.embed_sequences(sequences)
    predictions = embedder.logit_layer(embeddings)
    predicted_tokens = predictions.argmax(dim=-1)
    tokens = tokens[:, 1:-1]
    assert (tokens == predicted_tokens).sum() / tokens.numel() > 0.95
