import torch

from esm import pretrained

from netam.common import aa_strs_from_idx_tensor


def pad_embeddings(embeddings, desired_length):
    """
    Pads a batch of embeddings to a specified sequence length with zeros.

    Args:
        embeddings (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).
        desired_length (int): The length to which each sequence should be padded.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, desired_length, embedding_dim).
    """
    batch_size, seq_len, embedding_dim = embeddings.size()

    if desired_length <= 0:
        raise ValueError("desired_length must be a positive integer")

    # Truncate seq_len if it exceeds desired_length
    if seq_len > desired_length:
        embeddings = embeddings[:, :desired_length, :]
        seq_len = desired_length

    device = embeddings.device
    padded_embeddings = torch.zeros(
        (batch_size, desired_length, embedding_dim), device=device
    )
    padded_embeddings[:, :seq_len, :] = embeddings
    return padded_embeddings


class ESMEmbedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initializes the ESMEmbedder object.

        Args:
            model_name (str): Name of the pretrained ESM model (e.g., "esm2_t6_8M_UR50D").
            device (str): Device to run the model on.
        """
        self.device = device
        self.model, self.alphabet = pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(device)
        self.batch_converter = self.alphabet.get_batch_converter()

    @property
    def num_heads(self):
        return self.model.layers[0].self_attn.num_heads

    @property
    def d_model(self):
        return self.model.embed_dim

    @property
    def d_model_per_head(self):
        return self.d_model // self.num_heads

    @property
    def num_layers(self):
        return self.model.num_layers

    def embed_sequence_list(self, sequences: list[str]) -> torch.Tensor:
        """
        Embeds a batch of sequences.

        Args:
            sequences (list[str]): List of amino acid sequences.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, max_aa_seq_len, embedding_dim).
        """
        named_sequences = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(named_sequences)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.num_layers])
        embeddings = results["representations"][self.num_layers]

        return embeddings

    def embed_batch(self, amino_acid_indices: torch.Tensor) -> torch.Tensor:
        """
        Embeds a batch of netam amino acid indices.

        For now, we detokenize the amino acid indices and then use embed_sequence_list.

        Args:
            amino_acid_indices (torch.Tensor): A tensor of shape (batch_size, max_aa_seq_len).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, max_aa_seq_len, embedding_dim).
        """
        sequences = aa_strs_from_idx_tensor(amino_acid_indices)
        embedding = self.embed_sequence_list(sequences)
        desired_length = amino_acid_indices.size(1)
        padded_embedding = pad_embeddings(embedding, desired_length)
        return padded_embedding
