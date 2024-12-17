import pytest

from netam import protein_embedders
from netam.sequences import aa_idx_tensor_of_str


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
    assert tokens[0][0] == embedder.alphabet.cls_idx
    assert tokens[0][-1] == embedder.alphabet.eos_idx
    tokens = tokens[:, 1:-1]
    first_seq_idxs = aa_idx_tensor_of_str(sequences[0])
    translation_dict = {
        their_idx.item(): our_idx
        for our_idx, their_idx in enumerate(embedder.tok_to_aa_idxs)
    }
    for idx, token in zip(first_seq_idxs, tokens[0]):
        assert idx == translation_dict[token.item()]


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
